import tkinter as tk
from tkinter import ttk

from use_case_eval_core.ollama_client import requests
from use_case_eval_core.schemas import OLLAMA_TAGS_URL


CONNECT_ERROR_MESSAGE = (
    "Could not connect to Ollama at http://localhost:11434. Make sure Ollama is "
    "running, then refresh the model list."
)


class PlaceholderEntry(ttk.Entry):
    def __init__(self, master, placeholder, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.placeholder = placeholder
        self.default_foreground = ""
        self.placeholder_active = False
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._show_placeholder_if_empty)
        self.after_idle(self._initialize_placeholder)

    def _initialize_placeholder(self):
        self.default_foreground = self.cget("foreground")
        self._show_placeholder_if_empty()

    def _clear_placeholder(self, _event=None):
        if self.placeholder_active:
            self.delete(0, tk.END)
            self.configure(foreground=self.default_foreground)
            self.placeholder_active = False

    def _show_placeholder_if_empty(self, _event=None):
        if self.get():
            return
        self.placeholder_active = True
        self.configure(foreground="#777777")
        self.insert(0, self.placeholder)

    def value(self):
        if self.placeholder_active:
            return ""
        return self.get()


class PlaceholderText(tk.Text):
    def __init__(self, master, placeholder, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.placeholder = placeholder
        self.placeholder_active = False
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._show_placeholder_if_empty)
        self.after_idle(self._show_placeholder_if_empty)

    def _clear_placeholder(self, _event=None):
        if self.placeholder_active:
            self.delete("1.0", tk.END)
            self.configure(fg="#000000")
            self.placeholder_active = False

    def _show_placeholder_if_empty(self, _event=None):
        if self.get("1.0", "end-1c").strip():
            return
        self.placeholder_active = True
        self.configure(fg="#777777")
        self.delete("1.0", tk.END)
        self.insert("1.0", self.placeholder)

    def value(self):
        if self.placeholder_active:
            return ""
        return self.get("1.0", "end-1c")


class ScrollableCheckList(ttk.Frame):
    def __init__(self, master, height=150, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.variables = {}

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_inner)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_inner(self, event):
        self.canvas.itemconfigure(self.inner_window, width=event.width)

    def set_items(self, items, command=None):
        for child in self.inner.winfo_children():
            child.destroy()
        self.variables = {}

        if not items:
            ttk.Label(self.inner, text="No models available.").grid(
                row=0,
                column=0,
                sticky="w",
                padx=4,
                pady=4,
            )
            self._update_scroll_region()
            return

        for row_index, item in enumerate(items):
            variable = tk.BooleanVar(value=False)
            self.variables[item] = variable
            item_command = None
            if command is not None:
                item_command = lambda selected_item=item: command(selected_item)
            checkbox = ttk.Checkbutton(
                self.inner,
                text=item,
                variable=variable,
                command=item_command,
            )
            checkbox.grid(row=row_index, column=0, sticky="w", padx=4, pady=2)
        self._update_scroll_region()

    def selected(self):
        return [name for name, variable in self.variables.items() if variable.get()]


class UseCaseEvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Use Case Evaluator")
        self.model_names = []

        self._build_window()
        self.root.after(100, self.refresh_models)

    def _build_window(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(
            self.root,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.main_frame = ttk.Frame(self.canvas, padding=18)
        self.main_window = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.main_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.main_scrollbar.grid(row=0, column=1, sticky="ns")

        self.main_frame.bind("<Configure>", self._update_main_scroll_region)
        self.canvas.bind("<Configure>", self._resize_main_frame)

        self.main_frame.columnconfigure(0, weight=1)
        current_row = 0

        title = ttk.Label(
            self.main_frame,
            text="LLM Use Case Evaluator",
            font=("TkDefaultFont", 18, "bold"),
        )
        title.grid(row=current_row, column=0, sticky="w", pady=(0, 18))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="What use case would you like to evaluate:",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        self.use_case_entry = PlaceholderEntry(
            self.main_frame,
            "ex. Senior Voice Assistant",
        )
        self.use_case_entry.grid(row=current_row, column=0, sticky="ew", pady=(4, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text=(
                "Briefly describe the intended assistant in 1-4 sentences. Leave empty "
                "to continue without additional context:"
            ),
            wraplength=760,
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        self.context_text = PlaceholderText(
            self.main_frame,
            "ex. A personal assistant specialized for seniors.",
            height=5,
            wrap="word",
            relief="solid",
            borderwidth=1,
        )
        self.context_text.grid(row=current_row, column=0, sticky="ew", pady=(4, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="What Frontier Model would you like to use?",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        ttk.Label(
            self.main_frame,
            text="For best results, select the best model that your system can handle.",
            foreground="#555555",
            wraplength=760,
        ).grid(row=current_row, column=0, sticky="w", pady=(2, 4))
        current_row += 1
        self.frontier_model = tk.StringVar()
        self.frontier_dropdown = ttk.Combobox(
            self.main_frame,
            textvariable=self.frontier_model,
            state="readonly",
            values=[],
        )
        self.frontier_dropdown.grid(row=current_row, column=0, sticky="ew", pady=(0, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="What Models would you like to use for judging?",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        ttk.Label(
            self.main_frame,
            text="Select 1 or 2 of the best models that your system can handle.",
            foreground="#555555",
            wraplength=760,
        ).grid(row=current_row, column=0, sticky="w", pady=(2, 4))
        current_row += 1
        self.judge_checklist = ScrollableCheckList(self.main_frame, height=145)
        self.judge_checklist.grid(row=current_row, column=0, sticky="nsew", pady=(0, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="What Model(s) would you like to evaluate?",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        ttk.Label(
            self.main_frame,
            text="Select at least 1.",
            foreground="#555555",
            wraplength=760,
        ).grid(row=current_row, column=0, sticky="w", pady=(2, 4))
        current_row += 1
        self.evaluated_checklist = ScrollableCheckList(self.main_frame, height=175)
        self.evaluated_checklist.grid(row=current_row, column=0, sticky="nsew", pady=(0, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="How many sample questions would you like to run per model?",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        self.sample_questions = tk.StringVar(value="5")
        self.sample_dropdown = ttk.Combobox(
            self.main_frame,
            textvariable=self.sample_questions,
            state="readonly",
            values=[str(number) for number in range(1, 11)],
            width=8,
        )
        self.sample_dropdown.grid(row=current_row, column=0, sticky="w", pady=(4, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="Max tokens per model response:",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        self.max_tokens = tk.StringVar(value="220")
        validate_command = (self.root.register(self._validate_positive_integer), "%P")
        invalid_command = (self.root.register(self._reject_invalid_integer),)
        self.max_tokens_entry = ttk.Entry(
            self.main_frame,
            textvariable=self.max_tokens,
            validate="key",
            validatecommand=validate_command,
            invalidcommand=invalid_command,
            width=12,
        )
        self.max_tokens_entry.grid(row=current_row, column=0, sticky="w", pady=(4, 14))
        current_row += 1

        ttk.Label(
            self.main_frame,
            text="Judge Pass Threshold:",
        ).grid(row=current_row, column=0, sticky="w")
        current_row += 1
        self.judge_threshold = tk.StringVar(value="3")
        self.judge_threshold_dropdown = ttk.Combobox(
            self.main_frame,
            textvariable=self.judge_threshold,
            state="readonly",
            values=[str(number) for number in range(1, 6)],
            width=8,
        )
        self.judge_threshold_dropdown.grid(row=current_row, column=0, sticky="w", pady=(4, 14))
        current_row += 1

        self.refresh_button = ttk.Button(
            self.main_frame,
            text="Refresh Models",
            command=self.refresh_models,
        )
        self.refresh_button.grid(row=current_row, column=0, sticky="w", pady=(0, 10))
        current_row += 1

        self.status_message = tk.StringVar(value="")
        ttk.Label(
            self.main_frame,
            textvariable=self.status_message,
            foreground="#555555",
            wraplength=760,
        ).grid(row=current_row, column=0, sticky="ew", pady=(0, 14))
        current_row += 1

        self.run_button = ttk.Button(
            self.main_frame,
            text="Run Evaluation",
            state="disabled",
        )
        self.run_button.grid(row=current_row, column=0, sticky="ew")

    def _update_main_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_main_frame(self, event):
        self.canvas.itemconfigure(self.main_window, width=event.width)

    def _validate_positive_integer(self, proposed_value):
        if proposed_value == "":
            return True
        return proposed_value.isdigit() and int(proposed_value) > 0

    def _reject_invalid_integer(self):
        self.root.bell()
        self.status_message.set("Max tokens must be a positive integer.")

    def _fetch_ollama_models(self):
        if requests is None:
            raise ConnectionError(CONNECT_ERROR_MESSAGE)
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        model_names = set()
        for model in data.get("models", []):
            if not isinstance(model, dict):
                continue
            for key in ("name", "model"):
                model_name = model.get(key)
                if isinstance(model_name, str) and model_name.strip():
                    model_names.add(model_name.strip())
        return sorted(model_names, key=str.casefold)

    def refresh_models(self):
        self.status_message.set("Scanning Ollama for installed models...")
        self.root.update_idletasks()

        try:
            model_names = self._fetch_ollama_models()
        except Exception:
            self.model_names = []
            self._populate_model_controls([])
            self.status_message.set(CONNECT_ERROR_MESSAGE)
            return

        self.model_names = model_names
        self._populate_model_controls(model_names)

        if not model_names:
            self.status_message.set("No Ollama models were found.")
        else:
            self.status_message.set(f"Found {len(model_names)} Ollama model(s).")

    def _populate_model_controls(self, model_names):
        self.frontier_dropdown.configure(values=model_names)
        self.frontier_model.set(model_names[0] if model_names else "")
        self.judge_checklist.set_items(model_names, command=self._enforce_judge_limit)
        self.evaluated_checklist.set_items(model_names)

    def _enforce_judge_limit(self, changed_name):
        selected = self.judge_checklist.selected()
        if len(selected) <= 2:
            return

        self.judge_checklist.variables[changed_name].set(False)
        self.root.bell()
        self.status_message.set("Select no more than 2 judge models.")


def create_app():
    root = tk.Tk()
    root.minsize(720, 720)
    UseCaseEvalApp(root)
    return root


def main():
    root = create_app()
    root.mainloop()


if __name__ == "__main__":
    main()
