import ast
import builtins
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "posetoon_pipeline.py"
tree = ast.parse(open(path).read())

BUILTINS = set(dir(builtins))


class Scope(ast.NodeVisitor):
    def __init__(self):
        self.bound, self.loaded = set(), set()

    def visit_Name(self, n):
        (self.bound if isinstance(n.ctx, (ast.Store, ast.Del))
         else self.loaded).add(n.id)

    def visit_FunctionDef(self, n):
        self.bound.add(n.name)
        for a in n.args.args + n.args.kwonlyargs:
            self.bound.add(a.arg)
        if n.args.vararg:
            self.bound.add(n.args.vararg.arg)
        if n.args.kwarg:
            self.bound.add(n.args.kwarg.arg)
        self.generic_visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, n):
        for a in n.args.args:
            self.bound.add(a.arg)
        self.generic_visit(n)

    def visit_ClassDef(self, n):
        self.bound.add(n.name)
        self.generic_visit(n)

    def visit_comprehension(self, n):
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.bound.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Global(self, n):
        self.bound.update(n.names)


problems, n_cells = [], 0
exported, wanted = {}, {}
for node in tree.body:
    if not isinstance(node, ast.FunctionDef):
        continue
    if not any(isinstance(d, ast.Attribute) and d.attr == "cell"
               for d in node.decorator_list):
        continue
    n_cells += 1
    params = {a.arg for a in node.args.args}
    sc = Scope()
    for stmt in node.body:
        sc.visit(stmt)
    missing = sc.loaded - sc.bound - params - BUILTINS
    unused = params - sc.loaded
    if missing:
        problems.append((node.lineno, "MISSING", sorted(missing)))
    if unused:
        problems.append((node.lineno, "unused param", sorted(unused)))

    wanted.setdefault(node.lineno, set()).update(params)
    rets = [x for x in node.body if isinstance(x, ast.Return)]
    if rets and isinstance(rets[-1].value, ast.Tuple):
        exported[node.lineno] = {e.id for e in rets[-1].value.elts
                                 if isinstance(e, ast.Name)}
    elif rets and isinstance(rets[-1].value, ast.Name):
        exported[node.lineno] = {rets[-1].value.id}
    else:
        exported[node.lineno] = set()
    globals().setdefault("_binds", {})[node.lineno] = sc.bound

all_exported = set().union(*exported.values()) if exported else set()
all_wanted = set().union(*wanted.values()) if wanted else set()
not_exported = sorted(all_wanted - all_exported)

MIN_SUSPECT = 20
PER_FRAME_HINTS = ("kps", "series", "track", "table", "frames", "poses")
FRAME_ARG_CALLS = {"preview", "solve_pose", "render_frame"}


def _is_per_frame(name):
    n = name.lower()
    return any(h in n for h in PER_FRAME_HINTS)


frame_hits = []
for node in ast.walk(tree):
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        idx = node.slice
        if (isinstance(idx, ast.Constant) and isinstance(idx.value, int)
                and idx.value >= MIN_SUSPECT and _is_per_frame(node.value.id)):
            frame_hits.append((node.lineno, f"{node.value.id}[{idx.value}]"))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in FRAME_ARG_CALLS:
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, int) \
                    and a.value >= MIN_SUSPECT:
                frame_hits.append((node.lineno, f"{node.func.id}({a.value})"))
    if isinstance(node, ast.List):
        vals = [e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, int)]
        if vals and len(vals) == len(node.elts) and max(vals) >= MIN_SUSPECT \
                and len(vals) <= 6:
            frame_hits.append((node.lineno, f"literal list {vals}"))

print(f"scanned {n_cells} cells in {path}")
miss = [p for p in problems if p[1] == "MISSING"]
unus = [p for p in problems if p[1] != "MISSING"]
print(f"  cross-cell names MISSING from a signature: "
      f"{len(miss)}" + ("" if miss else "   <- OK"))
for line, _, names in miss:
    print(f"    line {line}: {names}")
print(f"  names in a signature but NEVER EXPORTED by any cell: "
      f"{len(not_exported)}" + ("" if not_exported else "   <- OK"))
for nm in not_exported:
    holders = [ln for ln, b in _binds.items() if nm in b]
    print(f"    {nm}  (bound in cell(s) at line {holders})")
print(f"  hardcoded frame numbers >= {MIN_SUSPECT} (check each is clamped): "
      f"{len(frame_hits)}" + ("" if frame_hits else "   <- OK"))
for line, what in frame_hits:
    print(f"    line {line}: {what}")
print(f"  unused params (harmless, but a signal an edit changed a cell): "
      f"{len(unus)}")
for line, _, names in unus:
    print(f"    line {line}: {names}")
