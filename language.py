"""
Define the language for Pauli Measurement based quantum computation.

Example:

qreg q[2];
creg c[10];
c[0] = MXX q[0], q[1];
if c[0]==1 then frame_X(q[0]);
c[1] = MY q[0];
"""


class Instruction:
    """Base class for all instructions."""
    def __init__(self, name, operands):
        self.name = name
        self.operands = operands

    def to_source(self) -> str:
        """
        Default pretty-printer: 'NAME op1, op2;'

        Subclasses override this when they need special syntax (like
        qreg/creg, measurement assignment, if-then, etc.).
        """
        if not self.operands:
            return f"{self.name};"

        op_strs = []
        for op in self.operands:
            # Use __str__ of RegisterRef / other objects when available.
            if isinstance(op, RegisterRef):
                op_strs.append(str(op))
            else:
                op_strs.append(str(op))
        return f"{self.name} " + ", ".join(op_strs) + ";"

    def __repr__(self) -> str:
        # Print in the DSL style when you do `print(instr)` or print a list.
        return self.to_source()


class RegisterRef:
    """Base class for a reference to a register element, e.g., q[0] or c[3]."""
    def __init__(self, reg_name, index):
        self.reg_name = reg_name
        self.index = index

    def __repr__(self):
        return f"{self.__class__.__name__}(reg_name={self.reg_name!r}, index={self.index})"

    def __str__(self):
        # When embedded in source code, show 'q[0]', 'c[3]', etc.
        return f"{self.reg_name}[{self.index}]"


class QubitRef(RegisterRef):
    """Reference to a quantum register element (qreg)."""
    pass


class CBitRef(RegisterRef):
    """Reference to a classical register element (creg)."""
    pass


class QRegDecl(Instruction):
    """Quantum register declaration: qreg q[2];"""

    def __init__(self, reg_name, size):
        super().__init__("qreg", [reg_name, size])
        self.reg_name = reg_name
        self.size = size

    def to_source(self) -> str:
        return f"qreg {self.reg_name}[{self.size}];"

    def __repr__(self) -> str:
        return self.to_source()


class CRegDecl(Instruction):
    """Classical register declaration: creg c[10];"""

    def __init__(self, reg_name, size):
        super().__init__("creg", [reg_name, size])
        self.reg_name = reg_name
        self.size = size

    def to_source(self) -> str:
        return f"creg {self.reg_name}[{self.size}];"

    def __repr__(self) -> str:
        return self.to_source()


class MeasurementInstruction(Instruction):
    """
    Measurement instruction in the Pauli measurement language.

    Examples:
        c[0] = MXX q[0], q[1];
        c[1] = MY q[0];
    """

    def __init__(self, name, qubits, classical_bit):
        # operands = [*qubits, classical_bit]
        super().__init__(name, qubits + [classical_bit])
        self.qubits = qubits               # list[QubitRef]
        self.classical_bit = classical_bit # CBitRef

    def to_source(self) -> str:
        if self.qubits:
            q_str = ", ".join(str(q) for q in self.qubits)
            return f"{self.classical_bit} = {self.name} {q_str};"
        else:
            return f"{self.classical_bit} = {self.name};"

    def __repr__(self) -> str:
        return self.to_source()


class FrameInstruction(Instruction):
    """
    Frame update instruction.

    Example:
        frame_X(q[0]);
    """

    def __init__(self, name, qubits):
        super().__init__(name, qubits)
        self.qubits = qubits  # list[QubitRef]

    def to_source(self) -> str:
        args = ", ".join(str(q) for q in self.qubits)
        return f"{self.name}({args});"

    def __repr__(self) -> str:
        return self.to_source()


class IfInstruction(Instruction):
    """
    Conditional instruction driven by a classical bit.

    Example:
        if c[0]==1 then frame_X(q[0]);
    """

    def __init__(self, cbit, value, body):
        # operands = [cbit, value, body]
        super().__init__("if", [cbit, value, body])
        self.cbit = cbit          # CBitRef
        self.value = value        # int (0 or 1)
        self.body = body          # Instruction

    def to_source(self) -> str:
        body_src = self.body.to_source().strip()
        # Strip trailing ';' so we don't end up with '... then frame_X(...);;'
        if body_src.endswith(";"):
            body_src = body_src[:-1]
        return f"if {self.cbit}=={self.value} then {body_src};"

    def __repr__(self) -> str:
        return self.to_source()
    

# ---------- Parsing helpers ----------

def _strip_trailing_semicolon(s: str) -> str:
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def _parse_reg_ref(s: str) -> RegisterRef:
    """
    Parse a register reference like 'q[0]' or 'c[3]'.

    The caller decides whether to interpret it as QubitRef or CBitRef,
    but we can infer from the variable naming convention as a first version.
    """
    s = s.strip()
    if "[" not in s or not s.endswith("]"):
        raise ValueError(f"Invalid register reference: {s!r}")

    reg_name, idx_part = s.split("[", 1)
    reg_name = reg_name.strip()
    idx_str = idx_part[:-1].strip()  # drop closing ']'
    index = int(idx_str)

    # Heuristic: q* -> QubitRef, c* -> CBitRef, otherwise base RegisterRef.
    if reg_name.startswith("q"):
        return QubitRef(reg_name, index)
    if reg_name.startswith("c"):
        return CBitRef(reg_name, index)
    return RegisterRef(reg_name, index)


def _parse_qubit_ref(s: str) -> QubitRef:
    ref = _parse_reg_ref(s)
    if isinstance(ref, QubitRef):
        return ref
    # For the first version, just coerce if needed.
    return QubitRef(ref.reg_name, ref.index)


def _parse_cbit_ref(s: str) -> CBitRef:
    ref = _parse_reg_ref(s)
    if isinstance(ref, CBitRef):
        return ref
    return CBitRef(ref.reg_name, ref.index)


def _parse_qreg_decl(line: str) -> QRegDecl:
    # line without trailing ';', starting after 'qreg'
    # e.g., 'q[2]' or 'q[2] // comment'
    rest = line[len("qreg"):].strip()
    # remove inline comments if any
    if "//" in rest:
        rest = rest.split("//", 1)[0].strip()
    if "[" not in rest or not rest.endswith("]"):
        raise ValueError(f"Invalid qreg declaration: {line!r}")
    name, size_part = rest.split("[", 1)
    name = name.strip()
    size_str = size_part[:-1].strip()
    size = int(size_str)
    return QRegDecl(name, size)


def _parse_creg_decl(line: str) -> CRegDecl:
    rest = line[len("creg"):].strip()
    if "//" in rest:
        rest = rest.split("//", 1)[0].strip()
    if "[" not in rest or not rest.endswith("]"):
        raise ValueError(f"Invalid creg declaration: {line!r}")
    name, size_part = rest.split("[", 1)
    name = name.strip()
    size_str = size_part[:-1].strip()
    size = int(size_str)
    return CRegDecl(name, size)


def _parse_frame_instruction(text: str) -> FrameInstruction:
    """
    Parse 'frame_X(q[0])' or similar.

    text should NOT include the trailing semicolon.
    """
    text = text.strip()
    # remove inline comments if any
    if "//" in text:
        text = text.split("//", 1)[0].strip()

    # Expect form: NAME(arglist)
    if "(" not in text or not text.endswith(")"):
        raise ValueError(f"Invalid frame instruction: {text!r}")

    name, arg_part = text.split("(", 1)
    name = name.strip()
    arg_str = arg_part[:-1].strip()  # drop ')'
    if not arg_str:
        qubits = []
    else:
        qubits = [_parse_qubit_ref(a) for a in arg_str.split(",")]

    return FrameInstruction(name, qubits)


def _parse_measurement_assignment(line: str) -> MeasurementInstruction:
    """
    Parse a measurement assignment like:
        c[0] = MXX q[0], q[1];
        c[1] = MY q[0];
    """
    line = _strip_trailing_semicolon(line)
    # remove inline comments if any
    if "//" in line:
        line = line.split("//", 1)[0].strip()

    if "=" not in line:
        raise ValueError(f"Not a measurement assignment: {line!r}")

    lhs, rhs = line.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    cbit = _parse_cbit_ref(lhs)

    # rhs: e.g. "MXX q[0], q[1]" or "MY q[0]"
    parts = rhs.split()
    if not parts:
        raise ValueError(f"Empty right-hand side in assignment: {line!r}")

    meas_name = parts[0].strip()
    operand_str = " ".join(parts[1:]).strip()

    if not operand_str:
        qubits = []
    else:
        # qubits are comma-separated
        qubit_tokens = [t.strip() for t in operand_str.split(",") if t.strip()]
        qubits = [_parse_qubit_ref(q) for q in qubit_tokens]

    return MeasurementInstruction(meas_name, qubits, cbit)


def _parse_if_instruction(line: str) -> IfInstruction:
    """
    Parse a conditional frame update like:
        if c[0]==1 then frame_X(q[0]);
    """
    line = _strip_trailing_semicolon(line)
    # remove leading 'if'
    if not line.startswith("if "):
        raise ValueError(f"Invalid if-instruction: {line!r}")

    rest = line[len("if "):].strip()
    # split on ' then '
    if " then " not in rest:
        raise ValueError(f"Missing 'then' in if-instruction: {line!r}")

    cond_str, body_str = rest.split(" then ", 1)
    cond_str = cond_str.strip()
    body_str = body_str.strip()

    # Parse condition: c[0]==1
    if "==" not in cond_str:
        raise ValueError(f"Invalid condition in if-instruction: {cond_str!r}")

    lhs, rhs = cond_str.split("==", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    cbit = _parse_cbit_ref(lhs)
    value = int(rhs)

    # For the first version, we assume the body is a single frame instruction.
    body_instr = _parse_frame_instruction(_strip_trailing_semicolon(body_str))

    return IfInstruction(cbit, value, body_instr)


# ---------- Public parsing API ----------

def parse_program(lines):
    """
    Parse a program given as a list of source lines.

    Supported (first version) constructs:
        qreg q[N];
        creg c[M];
        c[i] = MXX q[a], q[b];
        c[j] = MY q[a];
        if c[i]==1 then frame_X(q[a]);
    """
    instructions = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue

        # Strip trailing semicolon for classification (but keep original line
        # where necessary)
        line_no_semi = _strip_trailing_semicolon(line)

        if line_no_semi.startswith("qreg "):
            instr = _parse_qreg_decl(line_no_semi)
        elif line_no_semi.startswith("creg "):
            instr = _parse_creg_decl(line_no_semi)
        elif line_no_semi.startswith("if "):
            instr = _parse_if_instruction(line)
        elif "=" in line_no_semi:
            instr = _parse_measurement_assignment(line)
        elif line_no_semi.startswith("frame_"):
            instr = _parse_frame_instruction(line_no_semi)
        else:
            # Fallback: raw instruction if nothing above matches.
            parts = line_no_semi.split()
            name = parts[0]
            operands = parts[1:] if len(parts) > 1 else []
            instr = Instruction(name, operands)

        instructions.append(instr)

    return instructions


if __name__ == "__main__":
    # Minimal manual test on the example grammar.
    example_program = [
        "qreg q[2];",
        "creg c[10];",
        "c[0] = MXX q[0], q[1];",
        "if c[0]==1 then frame_X(q[0]);",
        "c[1] = MY q[0];",
    ]
    prog = parse_program(example_program)
    for instr in prog:
        print(instr)
