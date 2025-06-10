import numpy as np
from fractions import Fraction
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# --- CONFIGURAZIONE ---
console = Console()
np.set_printoptions(formatter={'object': lambda x: str(x)})


# --- CLASSE SIMPLEX ---
class Simplex:
    def __init__(self, objective_type, c, A, b, constraint_types, var_names):
        self.objective_type = objective_type;
        self.original_c = np.array(c, dtype=object)
        self.A = np.array(A, dtype=object);
        self.b = np.array(b, dtype=object)
        self.constraint_types = constraint_types;
        self.num_original_vars = len(c)
        self.num_constraints = len(b);
        self.original_var_names = var_names
        self.all_var_names = [];
        self.c = self.original_c.copy()
        self.is_maximization = False;
        self.slack_vars = []
        self.surplus_vars = [];
        self.artificial_vars = []
        ### MODIFICA: Mappa per tracciare l'origine delle variabili artificiali ###
        self.artificial_var_map = {}
        self.solution = None;
        self.optimal_value = None

    def _invert_matrix(self, matrix):
        n = len(matrix)
        if n == 0 or matrix.shape[0] != matrix.shape[1]: return None
        identity = np.array([[Fraction(0)] * n for _ in range(n)], dtype=object)
        for i in range(n): identity[i, i] = Fraction(1)
        aug = np.concatenate((matrix, identity), axis=1)
        for i in range(n):
            pivot_row = i
            while pivot_row < n and aug[pivot_row, i] == 0: pivot_row += 1
            if pivot_row == n: return None
            aug[[i, pivot_row]] = aug[[pivot_row, i]]
            pivot_val = aug[i, i]
            if pivot_val == 0: return None
            aug[i, :] /= pivot_val
            for j in range(n):
                if i != j:
                    factor = aug[j, i]
                    aug[j, :] -= factor * aug[i, :]
        return aug[:, n:]

    def _preprocess(self):
        console.print(Panel("[bold magenta]FASE DI PRE-ELABORAZIONE[/bold magenta]", expand=False))
        if self.objective_type == 'max':
            self.is_maximization = True;
            self.c *= -1
            console.print("Problema di massimizzazione convertito in minimizzazione (c * -1).")
        for i in range(self.num_constraints):
            if self.b[i] < 0:
                self.A[i] *= -1;
                self.b[i] *= -1
                if self.constraint_types[i] == '<=':
                    self.constraint_types[i] = '>='
                elif self.constraint_types[i] == '>=':
                    self.constraint_types[i] = '<='

        self.all_var_names = list(self.original_var_names)
        var_counter = self.num_original_vars + 1;
        art_counter = 1
        current_num_cols = self.A.shape[1]
        for i in range(self.num_constraints):
            ct = self.constraint_types[i]
            if ct == '<=':
                col = np.zeros((self.num_constraints, 1), dtype=object);
                col[i] = Fraction(1)
                self.A = np.c_[self.A, col];
                self.c = np.append(self.c, Fraction(0))
                self.slack_vars.append(current_num_cols)
                self.all_var_names.append(f"x{var_counter}");
                current_num_cols += 1;
                var_counter += 1
            elif ct == '>=':
                s_col = np.zeros((self.num_constraints, 1), dtype=object);
                s_col[i] = Fraction(-1)
                self.A = np.c_[self.A, s_col];
                self.c = np.append(self.c, Fraction(0))
                self.surplus_vars.append(current_num_cols)
                self.all_var_names.append(f"x{var_counter}");
                current_num_cols += 1;
                var_counter += 1
                a_col = np.zeros((self.num_constraints, 1), dtype=object);
                a_col[i] = Fraction(1)
                self.A = np.c_[self.A, a_col];
                self.c = np.append(self.c, Fraction(0))
                self.artificial_vars.append(current_num_cols)
                self.all_var_names.append(f"R{art_counter}")
                self.artificial_var_map[current_num_cols] = i  # Mappa l'indice della var con l'indice del vincolo
                current_num_cols += 1;
                art_counter += 1
            elif ct == '=':
                a_col = np.zeros((self.num_constraints, 1), dtype=object);
                a_col[i] = Fraction(1)
                self.A = np.c_[self.A, a_col];
                self.c = np.append(self.c, Fraction(0))
                self.artificial_vars.append(current_num_cols)
                self.all_var_names.append(f"R{art_counter}")
                self.artificial_var_map[current_num_cols] = i  # Mappa l'indice della var con l'indice del vincolo
                current_num_cols += 1;
                art_counter += 1
        console.print("[green]Problema convertito in forma standard.[/green]")

    def _simplex_iterator(self, c_objective, initial_basis):
        basis = list(initial_basis);
        iteration = 1
        while True:
            table = Table(title=f"Iterazione {iteration}", title_style="bold blue", show_header=True,
                          header_style="bold cyan")
            table.add_column("Elemento", style="dim", width=25);
            table.add_column("Valore")
            B = self.A[:, basis]
            B_inv = self._invert_matrix(B)
            if B_inv is None:
                console.print("[bold red]Errore: la matrice di base B è singolare.[/bold red]");
                return basis, None, None, None
            x_B = B_inv @ self.b;
            c_B = c_objective[basis];
            pi = c_B @ B_inv
            basis_names = [self.all_var_names[i] for i in basis]
            table.add_row("Variabili in base", str(basis_names))
            table.add_row("Valori variabili in base (x_B)", str(x_B))
            table.add_row("Vettore duale/pi (c_B * B_inv)", str(pi))
            reduced_costs = {};
            is_optimal = True
            for j in range(self.A.shape[1]):
                if j not in basis:
                    rc = (pi @ self.A[:, j]) - c_objective[j];
                    reduced_costs[j] = rc
                    if rc > 1e-9: is_optimal = False
            rc_str = ", ".join([f"{self.all_var_names[k]}: {v}" for k, v in reduced_costs.items()])
            table.add_row("Costi ridotti (z_j - c_j)", rc_str);
            console.print(table)
            if is_optimal:
                console.print("\n[bold green]Test di Ottimalità: SUPERATO.[/bold green]")
                final_solution = np.array([Fraction(0)] * self.A.shape[1], dtype=object);
                final_solution[basis] = x_B
                return basis, final_solution, c_B @ x_B, x_B
            entering_var = max(reduced_costs, key=reduced_costs.get)
            leaving_var_info, min_ratio = self._find_leaving_variable(d=B_inv @ self.A[:, entering_var], x_B=x_B,
                                                                      basis=basis)
            if leaving_var_info is None: return basis, None, -np.inf, None
            leaving_row_idx, leaving_var = leaving_var_info
            entering_var_name = self.all_var_names[entering_var];
            leaving_var_name = self.all_var_names[leaving_var]
            console.print(f"Variabile entrante (max z_j - c_j > 0)  : [bold yellow]{entering_var_name}[/bold yellow]")
            console.print(
                f"Test rapporto minimo (theta={min_ratio}) -> Variabile uscente: [bold red]{leaving_var_name}[/bold red]\n")
            basis[leaving_row_idx] = entering_var;
            basis.sort();
            iteration += 1

    def _find_leaving_variable(self, d, x_B, basis):
        min_ratio, leaving_row_idx = None, -1
        if np.all([val <= 0 for val in d]):
            console.print("\n[bold red]Test di Illimitatezza: SUPERATO.[/bold red]");
            return None, None
        for i in range(len(d)):
            if d[i] > 0:
                ratio = x_B[i] / d[i]
                if min_ratio is None or ratio < min_ratio:
                    min_ratio, leaving_row_idx = ratio, i
        if leaving_row_idx == -1:
            console.print("\n[bold red]Test di Illimitatezza: SUPERATO (nessun rapporto calcolabile).[/bold red]");
            return None, None
        leaving_var = basis[leaving_row_idx];
        return (leaving_row_idx, leaving_var), min_ratio

    def _phase1(self):
        title = "[bold magenta]INIZIO FASE 1[/bold magenta]\n[dim]Ricerca di una base ammissibile[/dim]"
        console.print(Panel(title, expand=False));
        c_phase1 = np.array([Fraction(0)] * self.A.shape[1], dtype=object)
        for var_idx in self.artificial_vars: c_phase1[var_idx] = Fraction(1)
        initial_basis = self.slack_vars + self.artificial_vars;
        initial_basis.sort()
        basis, _, value, x_B_final = self._simplex_iterator(c_phase1, initial_basis)

        if value > 1e-9:
            console.print("\n[bold red]RISULTATO FASE 1: Problema Inammissibile (costo > 0).[/bold red]");
            return None

        ### MODIFICA: Logica migliorata per la degenerazione ###
        rows_to_remove = []
        lingering_artificial_vars = []
        for i, var_idx in enumerate(basis):
            if var_idx in self.artificial_vars:
                if x_B_final[i] > 1e-9:
                    console.print(
                        f"\n[bold red]RISULTATO FASE 1: Inammissibile (var artificiale {self.all_var_names[var_idx]} > 0).[/bold red]");
                    return None
                else:
                    # È una var artificiale in base con valore 0. Segna la riga come ridondante.
                    rows_to_remove.append(self.artificial_var_map[var_idx])
                    lingering_artificial_vars.append(var_idx)

        console.print("\n[green]RISULTATO FASE 1: Base ammissibile trovata.[/green]")

        if rows_to_remove:
            console.print(
                f"[yellow]Attenzione: Trovati {len(rows_to_remove)} vincoli ridondanti. Verranno rimossi.[/yellow]")
            # Rimuovi le righe ridondanti da A e b
            self.A = np.delete(self.A, rows_to_remove, axis=0)
            self.b = np.delete(self.b, rows_to_remove, axis=0)
            self.num_constraints -= len(rows_to_remove)

        # Ricalcola la base per la Fase 2
        # Escludi TUTTE le variabili artificiali, sia quelle in base che non
        basis_after_removal = [b - sum(1 for art in self.artificial_vars if art < b) for b in basis if
                               b not in self.artificial_vars]

        # Rimuovi le colonne delle variabili artificiali
        self.all_var_names = [n for i, n in enumerate(self.all_var_names) if i not in self.artificial_vars]
        self.A = np.delete(self.A, self.artificial_vars, axis=1)
        temp_c = self.original_c.copy()
        if self.is_maximization: temp_c *= -1
        num_slack_surplus = len(self.slack_vars) + len(self.surplus_vars)
        self.c = np.append(temp_c, [Fraction(0)] * num_slack_surplus)
        console.print("Colonne delle variabili artificiali e righe ridondanti rimosse. Inizio Fase 2.")
        return basis_after_removal

    def _phase2(self, initial_basis):
        title = "[bold magenta]INIZIO FASE 2[/bold magenta]\n[dim]Ricerca della soluzione ottima[/dim]"
        console.print(Panel(title, expand=False))
        if not initial_basis:  # Se la base è vuota, non c'è niente da fare
            console.print("[yellow]La base per la Fase 2 è vuota. Nessuna soluzione da trovare.[/yellow]")
            self.solution = np.array([Fraction(0)] * self.num_original_vars)
            self.optimal_value = self.original_c @ self.solution
            return

        basis, solution, value, _ = self._simplex_iterator(self.c, initial_basis)
        if solution is None: return
        self.optimal_value = value
        if self.is_maximization: self.optimal_value *= -1
        self.solution = solution[:self.num_original_vars]

    def solve(self):
        self._preprocess()
        initial_basis = None
        if self.artificial_vars:
            initial_basis = self._phase1()
            if initial_basis is None:
                console.print(Panel("[bold red]Il problema originale è INAMMISSIBILE[/bold red]", expand=False,
                                    border_style="red"));
                return
        else:
            console.print("\n[green]Nessuna variabile artificiale necessaria.[/green]");
            initial_basis = self.slack_vars

        self._phase2(initial_basis)

        if self.solution is not None:
            title = "[bold green]SOLUZIONE OTTIMA TROVATA[/bold green]";
            border_style = "green"
            content = f"[bold]Tipo Problema:[/bold] {self.objective_type.upper()}\n"
            content += f"[bold]Valore Ottimo Z:[/bold] [yellow]{self.optimal_value}[/yellow]\n\n"
            content += "[bold]Valori delle variabili:[/bold]\n"
            for i in range(self.num_original_vars):
                if self.solution[i] > 0:  # Mostra solo le variabili con valore non nullo
                    content += f"  {self.original_var_names[i]} = {self.solution[i]}\n"
        else:
            title = "[bold red]SOLUZIONE NON TROVATA[/bold red]";
            border_style = "red"
            content = "Il problema non ha una soluzione ottima finita."
        console.print(Panel(content, title=title, expand=False, border_style=border_style))


# --- FUNZIONE DI INPUT (INVARIATA) ---
def get_problem_from_input_robust():
    console.print(Panel("[bold]Inserimento Dati del Problema di Programmazione Lineare[/bold]", expand=False))
    problem_data = {"obj_type": "min", "num_vars": 0, "num_constraints": 0, "var_names": [], "c": [], "A": [], "b": [],
                    "constraint_types": []}

    def get_type_and_counts():
        problem_data["obj_type"] = Prompt.ask("Tipo di problema", choices=["max", "min"], default="min")
        while True:
            try:
                problem_data["num_vars"] = int(Prompt.ask("Numero di variabili originali"))
                if problem_data["num_vars"] > 0:
                    break
                else:
                    console.print("[red]Il numero deve essere positivo.[/red]")
            except ValueError:
                console.print("[red]Input non valido.[/red]")
        while True:
            try:
                problem_data["num_constraints"] = int(Prompt.ask("Numero di vincoli"))
                if problem_data["num_constraints"] > 0:
                    break
                else:
                    console.print("[red]Il numero deve essere positivo.[/red]")
            except ValueError:
                console.print("[red]Input non valido.[/red]")

    def get_var_names():
        while True:
            prompt_text = "Nomi delle variabili (o Invio per default: x1, x2...)"
            var_names_str = Prompt.ask(f"[cyan]{prompt_text}[/cyan]")
            if not var_names_str:
                problem_data["var_names"] = [f"x{i + 1}" for i in range(problem_data["num_vars"])]
                console.print(f"Nomi di default: [dim]{' '.join(problem_data['var_names'])}[/dim]");
                break
            else:
                custom_names = var_names_str.split()
                if len(custom_names) == problem_data["num_vars"]:
                    problem_data["var_names"] = custom_names;
                    break
                else:
                    console.print(
                        f"[red]Errore: richiesti {problem_data['num_vars']} nomi, forniti {len(custom_names)}.[/red]")

    def get_objective():
        while True:
            c_str = Prompt.ask(f"Coefficienti funzione obiettivo ({problem_data['num_vars']} valori)")
            try:
                c = [Fraction(x) for x in c_str.split()]
                if len(c) == problem_data["num_vars"]:
                    problem_data["c"] = c; break
                else:
                    console.print(f"[red]Errore: richiesti {problem_data['num_vars']} coefficienti.[/red]")
            except (ValueError, ZeroDivisionError):
                console.print("[red]Input non valido. Inserire numeri, decimali o frazioni (es. 1/3).[/red]")

    def get_one_constraint(i):
        while True:
            constraint_str = Prompt.ask(f"Vincolo {i + 1}")
            try:
                parts = constraint_str.split();
                if len(parts) < 3: raise ValueError("Formato non valido.")
                op_str = parts[-2]
                if op_str not in ['<=', '>=', '=']: raise ValueError(f"Operatore '{op_str}' non valido.")
                b_val = Fraction(parts[-1]);
                a_row = [Fraction(x) for x in parts[:-2]]
                if len(a_row) != problem_data["num_vars"]: raise ValueError(
                    f"Richiesti {problem_data['num_vars']} coefficienti.")
                if len(problem_data["A"]) > i:
                    problem_data["A"][i], problem_data["b"][i], problem_data["constraint_types"][
                        i] = a_row, b_val, op_str
                else:
                    problem_data["A"].append(a_row);
                    problem_data["b"].append(b_val);
                    problem_data["constraint_types"].append(op_str)
                break
            except (ValueError, IndexError, ZeroDivisionError) as e:
                console.print(f"[red]Errore: {e}. Riprova.[/red]")

    def get_constraints():
        problem_data["A"], problem_data["b"], problem_data["constraint_types"] = [], [], []
        console.print(Panel("[bold]Inserisci i vincoli nel formato: `1/2 2 -3 <= 10`[/bold]", expand=False))
        for i in range(problem_data["num_constraints"]): get_one_constraint(i)

    def display_summary():
        def format_term(coeff, name):
            if coeff.numerator == 0: return ""
            sign = "+ " if coeff > 0 else "- "
            abs_coeff = abs(coeff)
            val_str = "" if abs_coeff == 1 else f"{abs_coeff} "
            return f"{sign}{val_str}{name}"

        obj_parts = [format_term(problem_data['c'][i], problem_data['var_names'][i]) for i in
                     range(problem_data['num_vars'])]
        obj_str = " ".join(filter(None, obj_parts)).lstrip('+ ')
        if not obj_str: obj_str = "0"
        summary = f"[bold]Tipo Problema:[/bold] {problem_data['obj_type'].upper()}\n"
        summary += f"[bold]Funzione Obiettivo:[/bold] Z = {obj_str}\n\n"
        summary += "[bold]Vincoli:[/bold]\n"
        for i in range(problem_data['num_constraints']):
            lhs_parts = [format_term(problem_data['A'][i][j], problem_data['var_names'][j]) for j in
                         range(problem_data['num_vars'])]
            lhs = " ".join(filter(None, lhs_parts)).lstrip('+ ')
            if not lhs: lhs = "0"
            summary += f"  {i + 1}: {lhs} {problem_data['constraint_types'][i]} {problem_data['b'][i]}\n"
        console.print(Panel(summary, title="Riepilogo del Problema", border_style="blue"))

    get_type_and_counts();
    get_var_names();
    get_objective();
    get_constraints()
    while True:
        display_summary()
        if Confirm.ask("I dati inseriti sono corretti?", default=True): break
        console.print("\n[bold yellow]Cosa desideri modificare?[/bold yellow]")
        choice = Prompt.ask("Scegli", choices=["1", "2", "3", "4", "5"],
                            prompt="[1] Tipo/Conteggi\n[2] Nomi Variabili\n[3] Funzione Obiettivo\n[4] Un Vincolo\n[5] Nessuna modifica")
        if choice == "1":
            get_type_and_counts(); get_var_names(); get_objective(); get_constraints()
        elif choice == "2":
            get_var_names()
        elif choice == "3":
            get_objective()
        elif choice == "4":
            while True:
                try:
                    idx = int(Prompt.ask(f"Quale vincolo? (1-{problem_data['num_constraints']})")) - 1
                    if 0 <= idx < problem_data['num_constraints']:
                        get_one_constraint(idx); break
                    else:
                        console.print(f"[red]Numero non valido.[/red]")
                except ValueError:
                    console.print("[red]Input non valido.[/red]")
        elif choice == "5":
            break
    return (problem_data["obj_type"], problem_data["c"], problem_data["A"],
            problem_data["b"], problem_data["constraint_types"], problem_data["var_names"])


if __name__ == "__main__":
    obj_type, c_vec, A_mat, b_vec, const_types, var_names = get_problem_from_input_robust()
    simplex_solver = Simplex(obj_type, c_vec, A_mat, b_vec, const_types, var_names)
    simplex_solver.solve()