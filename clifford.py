"""
Syntax for Clifford quantum instructions. We support the following gates:
- CNOT (Controlled NOT)
- H (Hadamard)
- S (Phase gate)
- X (Pauli-X)
- Y (Pauli-Y)
- Z (Pauli-Z)

Example:

qreg q[3];
CNOT q[0], q[1];
H q[2];
S q[1];
"""


# ------------ Core Data Structures ------------

class CliffordInstruction:
    """Base class for Clifford gate instructions."""

    def __init__(self, name, qubits, params=None):
        self.name = name
        self.qubits = qubits
        self.params = params if params is not None else []

    def __repr__(self):
        return (
            f"CliffordInstruction(name={self.name!r}, "
            f"qubits={self.qubits}, params={self.params})"
        )


class CNOT(CliffordInstruction):
    def __init__(self, control_qubit, target_qubit):
        super().__init__("CNOT", [control_qubit, target_qubit])


class SingleQubitGate(CliffordInstruction):
    """Generic single-qubit Clifford gate."""

    def __init__(self, name, qubit):
        super().__init__(name, [qubit])


# ------------ Register Reference Utilities ------------

class CliffordQubitRef:
    """Reference to a qubit such as q[0]."""

    def __init__(self, reg_name, index):
        self.reg_name = reg_name
        self.index = index

    def __repr__(self):
        return f"CliffordQubitRef(reg_name={self.reg_name!r}, index={self.index})"

    def __str__(self):
        return f"{self.reg_name}[{self.index}]"


class QRegDecl(CliffordInstruction):
    """Quantum register declaration: qreg q[3];"""

    def __init__(self, reg_name, size):
        super().__init__("qreg", [reg_name, size])
        self.reg_name = reg_name
        self.size = size

    def __repr__(self):
        return f"QRegDecl(reg_name={self.reg_name!r}, size={self.size})"


# ------------ Parsing Helpers ------------

def _strip_semi(s: str) -> str:
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def _parse_qubit_ref(token: str) -> CliffordQubitRef:
    token = token.strip()
    if "[" not in token or not token.endswith("]"):
        raise ValueError(f"Invalid qubit reference: {token!r}")

    name, idx_part = token.split("[", 1)
    name = name.strip()
    index = int(idx_part[:-1].strip())
    return CliffordQubitRef(name, index)


def _parse_qreg_decl(line: str) -> QRegDecl:
    # remove leading 'qreg'
    rest = line[len("qreg"):].strip()
    if "[" not in rest or not rest.endswith("]"):
        raise ValueError(f"Invalid qreg declaration: {line!r}")

    name, size_part = rest.split("[", 1)
    name = name.strip()
    size = int(size_part[:-1].strip())
    return QRegDecl(name, size)


# ------------ Gate Parsers ------------

_SINGLE_QUBIT_GATES = {"H", "S", "X", "Y", "Z"}


def _parse_single_qubit_gate(name: str, operand_str: str) -> SingleQubitGate:
    qubit = _parse_qubit_ref(operand_str)
    return SingleQubitGate(name, qubit)


def _parse_cnot_gate(operand_str: str) -> CNOT:
    parts = [p.strip() for p in operand_str.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"CNOT expects 2 operands, got: {operand_str!r}")

    control = _parse_qubit_ref(parts[0])
    target = _parse_qubit_ref(parts[1])
    return CNOT(control, target)


# ------------ Public Parsing API ------------

def parse_clifford_program(lines):
    """
    Parse a Clifford circuit program.

    Supported constructs:
        qreg q[N];
        CNOT q[a], q[b];
        H q[i];
        S q[i];
        X q[i];
        Y q[i];
        Z q[i];
    """

    instructions = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue

        no_semi = _strip_semi(line)

        # register declaration
        if no_semi.startswith("qreg "):
            instr = _parse_qreg_decl(no_semi)

        else:
            # first token is gate name
            parts = no_semi.split(maxsplit=1)
            name = parts[0]

            operands = parts[1] if len(parts) > 1 else ""

            if name == "CNOT":
                instr = _parse_cnot_gate(operands)

            elif name in _SINGLE_QUBIT_GATES:
                instr = _parse_single_qubit_gate(name, operands)

            else:
                raise ValueError(f"Unsupported Clifford instruction: {name!r}")

        instructions.append(instr)

    return instructions


# ------------ Minimal Test ------------

if __name__ == "__main__":
    example = [
        "qreg q[3];",
        "CNOT q[0], q[1];",
        "H q[2];",
        "S q[1];",
        "X q[0];",
    ]

    prog = parse_clifford_program(example)
    for instr in prog:
        print(instr)
