"""
Simplex Solver — Two-Phase Method
Implementazione del metodo del simplesso a due fasi con aritmetica esatta (frazioni).
"""

import numpy as np
import sympy as sp
from fractions import Fraction
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# ---------------------------------------------------------------------------
# Configurazione globale
# ---------------------------------------------------------------------------

console = Console()
np.set_printoptions(formatter={"object": lambda x: str(x)})

_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Eccezioni di dominio
# ---------------------------------------------------------------------------


class InfeasibleProblemError(Exception):
    """Sollevata quando il problema è inammissibile."""


class UnboundedProblemError(Exception):
    """Sollevata quando il problema è illimitato."""


class SingularBasisError(Exception):
    """Sollevata quando la matrice di base è singolare."""


# ---------------------------------------------------------------------------
# Utilità di algebra lineare
# ---------------------------------------------------------------------------


class LinearAlgebra:
    """Operazioni di algebra lineare con aritmetica esatta su oggetti Python."""

    @staticmethod
    def invert_matrix(matrix: np.ndarray) -> np.ndarray | None:
        sym_matrix = sp.Matrix(matrix.tolist())
        if sym_matrix.det() == 0:
            return None
        inv = sym_matrix.inv()
        return np.array(inv.tolist(), dtype=object)

# ---------------------------------------------------------------------------
# Struttura dati del problema
# ---------------------------------------------------------------------------


class LinearProgrammingProblem:
    """Rappresenta i dati grezzi di un problema di programmazione lineare."""

    def __init__(
        self,
        objective_type: str,
        c: list,
        A: list,
        b: list,
        constraint_types: list[str],
        var_names: list[str],
    ) -> None:
        self.objective_type = objective_type
        self.c = np.array(c, dtype=object)
        self.A = np.array(A, dtype=object)
        self.b = np.array(b, dtype=object)
        self.constraint_types = list(constraint_types)
        self.var_names = list(var_names)

    @property
    def num_vars(self) -> int:
        return len(self.c)

    @property
    def num_constraints(self) -> int:
        return len(self.b)


# ---------------------------------------------------------------------------
# Iteratore del simplesso (nucleo algoritmico)
# ---------------------------------------------------------------------------


class SimplexIterator:
    """
    Esegue le iterazioni del simplesso su un tableau dato.
    Separato dalla logica di costruzione del problema per chiarezza.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, var_names: list[str]) -> None:
        self._A = A
        self._b = b
        self._var_names = var_names

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def run(
        self, c_objective: np.ndarray, initial_basis: list[int]
    ) -> tuple[list[int], np.ndarray | None, object | None, np.ndarray | None]:
        """
        Esegue il simplesso a partire da una base iniziale ammissibile.

        Returns:
            (basis, solution, optimal_value, x_B_final)
            solution è None se il problema è illimitato o la base è singolare.
        """
        basis = list(initial_basis)
        iteration = 1

        while True:
            console.print(self._build_iteration_table(c_objective, basis, iteration))

            B_inv, x_B, pi = self._compute_basis_quantities(c_objective, basis)
            reduced_costs = self._compute_reduced_costs(c_objective, pi)

            self._print_reduced_costs(reduced_costs)

            if self._is_optimal(reduced_costs):
                console.print("\n[bold green]Test di Ottimalità: SUPERATO.[/bold green]")
                solution = self._build_solution(basis, x_B)
                optimal_value = c_objective[basis] @ x_B
                return basis, solution, optimal_value, x_B

            entering_var = self._select_entering_variable(reduced_costs)
            direction = B_inv @ self._A[:, entering_var]

            leaving_info, theta = self._find_leaving_variable(direction, x_B, basis)
            if leaving_info is None:
                return basis, None, -np.inf, None

            leaving_row_idx, leaving_var = leaving_info
            self._print_pivot_info(entering_var, leaving_var, theta)

            basis[leaving_row_idx] = entering_var
            basis.sort()
            iteration += 1

    # ------------------------------------------------------------------
    # Calcoli del tableau
    # ------------------------------------------------------------------

    def _compute_basis_quantities(
        self, c_objective: np.ndarray, basis: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = self._A[:, basis]
        B_inv = LinearAlgebra.invert_matrix(B)
        if B_inv is None:
            raise SingularBasisError("La matrice di base B è singolare.")
        x_B = B_inv @ self._b
        c_B = c_objective[basis]
        pi = c_B @ B_inv
        return B_inv, x_B, pi

    def _compute_reduced_costs(
        self, c_objective: np.ndarray, pi: np.ndarray
    ) -> dict[int, object]:
        return {
            j: (pi @ self._A[:, j]) - c_objective[j]
            for j in range(self._A.shape[1])
        }

    def _is_optimal(self, reduced_costs: dict[int, object]) -> bool:
        return all(rc <= _EPSILON for rc in reduced_costs.values())

    def _select_entering_variable(self, reduced_costs: dict[int, object]) -> int:
        return max(reduced_costs, key=reduced_costs.get)

    def _find_leaving_variable(
        self, direction: np.ndarray, x_B: np.ndarray, basis: list[int]
    ) -> tuple[tuple[int, int] | None, object | None]:
        positive_indices = [i for i, d in enumerate(direction) if d > 0]
        if not positive_indices:
            console.print("\n[bold red]Test di Illimitatezza: SUPERATO.[/bold red]")
            return None, None

        min_ratio, leaving_row = None, -1
        for i in positive_indices:
            ratio = x_B[i] / direction[i]
            if min_ratio is None or ratio < min_ratio:
                min_ratio, leaving_row = ratio, i

        return (leaving_row, basis[leaving_row]), min_ratio

    def _build_solution(self, basis: list[int], x_B: np.ndarray) -> np.ndarray:
        solution = np.array([Fraction(0)] * self._A.shape[1], dtype=object)
        solution[basis] = x_B
        return solution

    # ------------------------------------------------------------------
    # Presentazione
    # ------------------------------------------------------------------

    def _build_iteration_table(
        self, c_objective: np.ndarray, basis: list[int], iteration: int
    ) -> Table:
        B = self._A[:, basis]
        B_inv = LinearAlgebra.invert_matrix(B)
        x_B = B_inv @ self._b
        c_B = c_objective[basis]
        pi = c_B @ B_inv

        table = Table(
            title=f"Iterazione {iteration}",
            title_style="bold blue",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Elemento", style="dim", width=25)
        table.add_column("Valore")

        basis_names = [self._var_names[i] for i in basis]
        table.add_row("Variabili in base", str(basis_names))
        table.add_row("Valori variabili in base (x_B)", str(x_B))
        table.add_row("Vettore duale/pi (c_B * B_inv)", str(pi))
        return table

    def _print_reduced_costs(self, reduced_costs: dict[int, object]) -> None:
        rc_str = ", ".join(
            f"{self._var_names[k]}: {v}" for k, v in reduced_costs.items()
        )
        console.print(f"Costi ridotti (z_j - c_j): {rc_str}")

    def _print_pivot_info(
        self, entering_var: int, leaving_var: int, theta: object
    ) -> None:
        entering_name = self._var_names[entering_var]
        leaving_name = self._var_names[leaving_var]
        console.print(
            f"Variabile entrante (max z_j - c_j > 0)  : [bold yellow]{entering_name}[/bold yellow]"
        )
        console.print(
            f"Test rapporto minimo (theta={theta}) -> Variabile uscente: "
            f"[bold red]{leaving_name}[/bold red]\n"
        )


# ---------------------------------------------------------------------------
# Preprocessore: costruisce la forma standard
# ---------------------------------------------------------------------------


class StandardFormBuilder:
    """
    Trasforma un LinearProgrammingProblem nella forma standard aggiungendo
    variabili di slack, surplus e artificiali.
    """

    def __init__(self, problem: LinearProgrammingProblem) -> None:
        self._problem = problem

        # Copie modificabili
        self.A = problem.A.copy()
        self.b = problem.b.copy()
        self.constraint_types = list(problem.constraint_types)
        self.c = problem.c.copy()
        self.is_maximization = problem.objective_type == "max"

        self.all_var_names: list[str] = []
        self.slack_vars: list[int] = []
        self.surplus_vars: list[int] = []
        self.artificial_vars: list[int] = []
        self.artificial_var_map: dict[int, int] = {}  # var_index -> constraint_index

    def build(self) -> None:
        """Esegue la trasformazione in forma standard."""
        console.print(Panel("[bold magenta]FASE DI PRE-ELABORAZIONE[/bold magenta]", expand=False))
        self._negate_objective_if_maximization()
        self._ensure_nonnegative_rhs()
        self._add_auxiliary_variables()
        console.print("[green]Problema convertito in forma standard.[/green]")

    # ------------------------------------------------------------------
    # Fasi di pre-elaborazione
    # ------------------------------------------------------------------

    def _negate_objective_if_maximization(self) -> None:
        if self.is_maximization:
            self.c *= -1
            console.print("Problema di massimizzazione convertito in minimizzazione (c * -1).")

    def _ensure_nonnegative_rhs(self) -> None:
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i] *= -1
                self.b[i] *= -1
                if self.constraint_types[i] == "<=":
                    self.constraint_types[i] = ">="
                elif self.constraint_types[i] == ">=":
                    self.constraint_types[i] = "<="

    def _add_auxiliary_variables(self) -> None:
        self.all_var_names = list(self._problem.var_names)
        num_constraints = len(self.b)
        col_idx = self.A.shape[1]
        var_counter = col_idx + 1
        art_counter = 1

        for i, ct in enumerate(self.constraint_types):
            if ct == "<=":
                col_idx, var_counter = self._add_slack(i, num_constraints, col_idx, var_counter)
            elif ct == ">=":
                col_idx, var_counter = self._add_surplus(i, num_constraints, col_idx, var_counter)
                col_idx, var_counter, art_counter = self._add_artificial(
                    i, num_constraints, col_idx, var_counter, art_counter
                )
            elif ct == "=":
                col_idx, var_counter, art_counter = self._add_artificial(
                    i, num_constraints, col_idx, var_counter, art_counter
                )

    def _add_slack(
        self, row: int, n_rows: int, col_idx: int, var_counter: int
    ) -> tuple[int, int]:
        col = self._unit_column(n_rows, row)
        self.A = np.c_[self.A, col]
        self.c = np.append(self.c, Fraction(0))
        self.slack_vars.append(col_idx)
        self.all_var_names.append(f"x{var_counter}")
        return col_idx + 1, var_counter + 1

    def _add_surplus(
        self, row: int, n_rows: int, col_idx: int, var_counter: int
    ) -> tuple[int, int]:
        col = self._unit_column(n_rows, row, coefficient=Fraction(-1))
        self.A = np.c_[self.A, col]
        self.c = np.append(self.c, Fraction(0))
        self.surplus_vars.append(col_idx)
        self.all_var_names.append(f"x{var_counter}")
        return col_idx + 1, var_counter + 1

    def _add_artificial(
        self, row: int, n_rows: int, col_idx: int, var_counter: int, art_counter: int
    ) -> tuple[int, int, int]:
        col = self._unit_column(n_rows, row)
        self.A = np.c_[self.A, col]
        self.c = np.append(self.c, Fraction(0))
        self.artificial_vars.append(col_idx)
        self.artificial_var_map[col_idx] = row
        self.all_var_names.append(f"R{art_counter}")
        return col_idx + 1, var_counter + 1, art_counter + 1

    @staticmethod
    def _unit_column(
        n_rows: int, one_at: int, coefficient: Fraction = Fraction(1)
    ) -> np.ndarray:
        col = np.zeros((n_rows, 1), dtype=object)
        col[one_at] = coefficient
        return col


# ---------------------------------------------------------------------------
# Metodo dei due fasi
# ---------------------------------------------------------------------------


class TwoPhaseSimplex:
    """
    Implementa il metodo del simplesso a due fasi.
    Coordina StandardFormBuilder e SimplexIterator senza contenere
    logica algoritmca diretta.
    """

    def __init__(self, problem: LinearProgrammingProblem) -> None:
        self._problem = problem
        self._builder = StandardFormBuilder(problem)

        # Risultati esposti dopo solve()
        self.solution: np.ndarray | None = None
        self.optimal_value: object | None = None

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Punto di ingresso principale: pre-elabora e risolve il problema."""
        self._builder.build()

        try:
            initial_basis = self._determine_initial_basis()
            self._run_phase2(initial_basis)
        except InfeasibleProblemError:
            console.print(
                Panel(
                    "[bold red]Il problema originale è INAMMISSIBILE[/bold red]",
                    expand=False,
                    border_style="red",
                )
            )
            return

        self._print_result()

    # ------------------------------------------------------------------
    # Determinazione della base iniziale
    # ------------------------------------------------------------------

    def _determine_initial_basis(self) -> list[int]:
        if not self._builder.artificial_vars:
            console.print("\n[green]Nessuna variabile artificiale necessaria.[/green]")
            return list(self._builder.slack_vars)
        return self._run_phase1()

    def _run_phase1(self) -> list[int]:
        console.print(
            Panel(
                "[bold magenta]INIZIO FASE 1[/bold magenta]\n[dim]Ricerca di una base ammissibile[/dim]",
                expand=False,
            )
        )

        c_phase1 = self._build_phase1_objective()
        initial_basis = sorted(self._builder.slack_vars + self._builder.artificial_vars)

        iterator = SimplexIterator(
            self._builder.A, self._builder.b, self._builder.all_var_names
        )
        basis, _, value, x_B_final = iterator.run(c_phase1, initial_basis)

        if value > _EPSILON:
            console.print("\n[bold red]RISULTATO FASE 1: Problema Inammissibile (costo > 0).[/bold red]")
            raise InfeasibleProblemError()

        basis = self._handle_degenerate_artificials(basis, x_B_final)
        self._strip_artificial_variables()

        console.print("[green]Colonne delle variabili artificiali e righe ridondanti rimosse. Inizio Fase 2.[/green]")
        return basis

    def _build_phase1_objective(self) -> np.ndarray:
        c_phase1 = np.array([Fraction(0)] * self._builder.A.shape[1], dtype=object)
        for var_idx in self._builder.artificial_vars:
            c_phase1[var_idx] = Fraction(1)
        return c_phase1

    def _handle_degenerate_artificials(
        self, basis: list[int], x_B_final: np.ndarray
    ) -> list[int]:
        """
        Gestisce le variabili artificiali rimaste in base con valore zero
        (degenerazione): rimuove i vincoli ridondanti corrispondenti.
        """
        rows_to_remove = []
        for i, var_idx in enumerate(basis):
            if var_idx not in self._builder.artificial_vars:
                continue
            if x_B_final[i] > _EPSILON:
                console.print(
                    f"\n[bold red]RISULTATO FASE 1: Inammissibile "
                    f"(var artificiale {self._builder.all_var_names[var_idx]} > 0).[/bold red]"
                )
                raise InfeasibleProblemError()
            rows_to_remove.append(self._builder.artificial_var_map[var_idx])

        console.print("\n[green]RISULTATO FASE 1: Base ammissibile trovata.[/green]")

        if rows_to_remove:
            console.print(
                f"[yellow]Attenzione: Trovati {len(rows_to_remove)} vincoli ridondanti. "
                f"Verranno rimossi.[/yellow]"
            )
            self._builder.A = np.delete(self._builder.A, rows_to_remove, axis=0)
            self._builder.b = np.delete(self._builder.b, rows_to_remove, axis=0)

        # Ricalcola gli indici della base escludendo le variabili artificiali
        arts = set(self._builder.artificial_vars)
        return [
            b - sum(1 for art in self._builder.artificial_vars if art < b)
            for b in basis
            if b not in arts
        ]

    def _strip_artificial_variables(self) -> None:
        """Rimuove colonne artificiali da A, c e aggiorna i nomi delle variabili."""
        arts = self._builder.artificial_vars
        self._builder.all_var_names = [
            n for i, n in enumerate(self._builder.all_var_names) if i not in arts
        ]
        self._builder.A = np.delete(self._builder.A, arts, axis=1)

        # Ricostruisce c per la Fase 2
        original_c = self._problem.c.copy()
        if self._builder.is_maximization:
            original_c *= -1
        num_slack_surplus = len(self._builder.slack_vars) + len(self._builder.surplus_vars)
        self._builder.c = np.append(original_c, [Fraction(0)] * num_slack_surplus)

    # ------------------------------------------------------------------
    # Fase 2
    # ------------------------------------------------------------------

    def _run_phase2(self, initial_basis: list[int]) -> None:
        console.print(
            Panel(
                "[bold magenta]INIZIO FASE 2[/bold magenta]\n[dim]Ricerca della soluzione ottima[/dim]",
                expand=False,
            )
        )

        if not initial_basis:
            console.print("[yellow]La base per la Fase 2 è vuota. Nessuna soluzione da trovare.[/yellow]")
            self.solution = np.array([Fraction(0)] * self._problem.num_vars)
            self.optimal_value = self._problem.c @ self.solution
            return

        iterator = SimplexIterator(
            self._builder.A, self._builder.b, self._builder.all_var_names
        )
        _, solution, value, _ = iterator.run(self._builder.c, initial_basis)

        if solution is None:
            return

        self.optimal_value = value * (-1 if self._builder.is_maximization else 1)
        self.solution = solution[: self._problem.num_vars]

    # ------------------------------------------------------------------
    # Output del risultato
    # ------------------------------------------------------------------

    def _print_result(self) -> None:
        if self.solution is not None:
            self._print_optimal_solution()
        else:
            console.print(
                Panel(
                    "Il problema non ha una soluzione ottima finita.",
                    title="[bold red]SOLUZIONE NON TROVATA[/bold red]",
                    expand=False,
                    border_style="red",
                )
            )

    def _print_optimal_solution(self) -> None:
        content = (
            f"[bold]Tipo Problema:[/bold] {self._problem.objective_type.upper()}\n"
            f"[bold]Valore Ottimo Z:[/bold] [yellow]{self.optimal_value}[/yellow]\n\n"
            "[bold]Valori delle variabili:[/bold]\n"
        )
        for i in range(self._problem.num_vars):
            if self.solution[i] > 0:
                content += f"  {self._problem.var_names[i]} = {self.solution[i]}\n"

        console.print(
            Panel(
                content,
                title="[bold green]SOLUZIONE OTTIMA TROVATA[/bold green]",
                expand=False,
                border_style="green",
            )
        )


# ---------------------------------------------------------------------------
# Input interattivo
# ---------------------------------------------------------------------------


class ProblemInputHandler:
    """Gestisce l'acquisizione interattiva del problema dall'utente."""

    def __init__(self) -> None:
        self._data: dict = {
            "obj_type": "min",
            "num_vars": 0,
            "num_constraints": 0,
            "var_names": [],
            "c": [],
            "A": [],
            "b": [],
            "constraint_types": [],
        }

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def collect(self) -> LinearProgrammingProblem:
        """Guida l'utente nell'inserimento del problema e restituisce un LP."""
        console.print(
            Panel("[bold]Inserimento Dati del Problema di Programmazione Lineare[/bold]", expand=False)
        )
        self._get_type_and_counts()
        self._get_var_names()
        self._get_objective()
        self._get_all_constraints()

        while True:
            self._display_summary()
            if Confirm.ask("I dati inseriti sono corretti?", default=True):
                break
            self._handle_correction()

        d = self._data
        return LinearProgrammingProblem(
            objective_type=d["obj_type"],
            c=d["c"],
            A=d["A"],
            b=d["b"],
            constraint_types=d["constraint_types"],
            var_names=d["var_names"],
        )

    # ------------------------------------------------------------------
    # Raccolta dei dati
    # ------------------------------------------------------------------

    def _get_type_and_counts(self) -> None:
        self._data["obj_type"] = Prompt.ask("Tipo di problema", choices=["max", "min"], default="min")
        self._data["num_vars"] = self._ask_positive_int("Numero di variabili originali")
        self._data["num_constraints"] = self._ask_positive_int("Numero di vincoli")

    def _get_var_names(self) -> None:
        n = self._data["num_vars"]
        while True:
            raw = Prompt.ask("[cyan]Nomi delle variabili (o Invio per default: x1, x2...)[/cyan]")
            if not raw:
                self._data["var_names"] = [f"x{i + 1}" for i in range(n)]
                console.print(f"Nomi di default: [dim]{' '.join(self._data['var_names'])}[/dim]")
                return
            names = raw.split()
            if len(names) == n:
                self._data["var_names"] = names
                return
            console.print(f"[red]Errore: richiesti {n} nomi, forniti {len(names)}.[/red]")

    def _get_objective(self) -> None:
        n = self._data["num_vars"]
        while True:
            raw = Prompt.ask(f"Coefficienti funzione obiettivo ({n} valori)")
            try:
                coeffs = [Fraction(x) for x in raw.split()]
                if len(coeffs) == n:
                    self._data["c"] = coeffs
                    return
                console.print(f"[red]Errore: richiesti {n} coefficienti.[/red]")
            except (ValueError, ZeroDivisionError):
                console.print("[red]Input non valido. Inserire numeri, decimali o frazioni (es. 1/3).[/red]")

    def _get_all_constraints(self) -> None:
        self._data["A"] = []
        self._data["b"] = []
        self._data["constraint_types"] = []
        console.print(Panel("[bold]Inserisci i vincoli nel formato: `1/2 2 -3 <= 10`[/bold]", expand=False))
        for i in range(self._data["num_constraints"]):
            self._get_one_constraint(i)

    def _get_one_constraint(self, idx: int) -> None:
        n = self._data["num_vars"]
        while True:
            raw = Prompt.ask(f"Vincolo {idx + 1}")
            try:
                parts = raw.split()
                if len(parts) < 3:
                    raise ValueError("Formato non valido.")
                operator = parts[-2]
                if operator not in ("<=", ">=", "="):
                    raise ValueError(f"Operatore '{operator}' non valido.")
                rhs = Fraction(parts[-1])
                lhs = [Fraction(x) for x in parts[:-2]]
                if len(lhs) != n:
                    raise ValueError(f"Richiesti {n} coefficienti.")

                # Aggiorna o appende
                if idx < len(self._data["A"]):
                    self._data["A"][idx] = lhs
                    self._data["b"][idx] = rhs
                    self._data["constraint_types"][idx] = operator
                else:
                    self._data["A"].append(lhs)
                    self._data["b"].append(rhs)
                    self._data["constraint_types"].append(operator)
                return
            except (ValueError, IndexError, ZeroDivisionError) as exc:
                console.print(f"[red]Errore: {exc}. Riprova.[/red]")

    # ------------------------------------------------------------------
    # Gestione correzioni
    # ------------------------------------------------------------------

    def _handle_correction(self) -> None:
        console.print("\n[bold yellow]Cosa desideri modificare?[/bold yellow]")
        choice = Prompt.ask(
            "Scegli",
            choices=["1", "2", "3", "4", "5"],
            prompt=(
                "[1] Tipo/Conteggi\n"
                "[2] Nomi Variabili\n"
                "[3] Funzione Obiettivo\n"
                "[4] Un Vincolo\n"
                "[5] Nessuna modifica"
            ),
        )
        if choice == "1":
            self._get_type_and_counts()
            self._get_var_names()
            self._get_objective()
            self._get_all_constraints()
        elif choice == "2":
            self._get_var_names()
        elif choice == "3":
            self._get_objective()
        elif choice == "4":
            self._ask_and_edit_constraint()

    def _ask_and_edit_constraint(self) -> None:
        n = self._data["num_constraints"]
        while True:
            try:
                idx = int(Prompt.ask(f"Quale vincolo? (1-{n})")) - 1
                if 0 <= idx < n:
                    self._get_one_constraint(idx)
                    return
                console.print("[red]Numero non valido.[/red]")
            except ValueError:
                console.print("[red]Input non valido.[/red]")

    # ------------------------------------------------------------------
    # Visualizzazione riepilogo
    # ------------------------------------------------------------------

    def _display_summary(self) -> None:
        d = self._data
        obj_str = self._format_linear_expression(d["c"], d["var_names"]) or "0"
        summary = (
            f"[bold]Tipo Problema:[/bold] {d['obj_type'].upper()}\n"
            f"[bold]Funzione Obiettivo:[/bold] Z = {obj_str}\n\n"
            "[bold]Vincoli:[/bold]\n"
        )
        for i in range(d["num_constraints"]):
            lhs = self._format_linear_expression(d["A"][i], d["var_names"]) or "0"
            summary += f"  {i + 1}: {lhs} {d['constraint_types'][i]} {d['b'][i]}\n"

        console.print(Panel(summary, title="Riepilogo del Problema", border_style="blue"))

    @staticmethod
    def _format_linear_expression(coefficients: list, names: list[str]) -> str:
        terms = []
        for coeff, name in zip(coefficients, names):
            if coeff == 0:
                continue
            sign = "+ " if coeff > 0 else "- "
            abs_val = abs(coeff)
            val_str = "" if abs_val == 1 else f"{abs_val} "
            terms.append(f"{sign}{val_str}{name}")
        return " ".join(terms).lstrip("+ ")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ask_positive_int(prompt: str) -> int:
        while True:
            try:
                value = int(Prompt.ask(prompt))
                if value > 0:
                    return value
                console.print("[red]Il numero deve essere positivo.[/red]")
            except ValueError:
                console.print("[red]Input non valido.[/red]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    handler = ProblemInputHandler()
    problem = handler.collect()
    solver = TwoPhaseSimplex(problem)
    solver.solve()


if __name__ == "__main__":
    main()