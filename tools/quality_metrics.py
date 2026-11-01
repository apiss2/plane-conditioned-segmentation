"""Report whole-src coverage and function-local Radon CC/CRAP without scope exclusions."""

import argparse
import ast
import dis
import hashlib
import json
import types
from pathlib import Path

import coverage
from radon.visitors import ComplexityVisitor


def function_rows(source_path, coverage_file, report, text_override=None, display_name=None):
    text = text_override if text_override is not None else source_path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    definitions = {}

    def visit(node, parents=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = '.'.join((*parents, node.name))
            if key in definitions:
                raise ValueError('Duplicate qualified function definition requires manual analysis: ' + key)
            definitions[key] = node
            parents = (*parents, node.name)
        elif isinstance(node, ast.ClassDef):
            parents = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree)
    compiled = compile(text, str(source_path), 'exec')
    codes = {}
    anonymous = []

    def collect(code, owner=None):
        key = code.co_qualname.replace('.<locals>.', '.')
        if key in definitions:
            owner = key
            codes[key] = [code]
        elif code.co_name.startswith('<') and code.co_name != '<module>' and owner:
            codes[owner].append(code)
        if code.co_name.startswith('<') and code.co_name != '<module>':
            anonymous.append(code)
        for item in code.co_consts:
            if isinstance(item, types.CodeType):
                collect(item, owner)

    collect(compiled)
    cov = coverage.Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()
    measured = next((f for f in data.measured_files() if Path(f).resolve() == source_path.resolve()), None)
    arcs = set(data.arcs(measured) or ()) if measured else set()
    statements = set(report['executed_lines']) | set(report['missing_lines'])
    executed = set(report['executed_lines'])
    rows = []
    for name, node in definitions.items():
        local = codes[name]
        own = set()
        for code in local:
            for instruction in dis.get_instructions(code):
                if (
                    instruction.opname not in ['RESUME', 'RETURN_GENERATOR', 'COPY_FREE_VARS', 'CACHE', 'EXTENDED_ARG']
                    and instruction.positions.lineno
                ):
                    own.add(instruction.positions.lineno)
        own &= statements
        first = local[0].co_firstlineno
        collision = [c.co_qualname for c in anonymous if c.co_firstlineno == first and c not in local]
        entered = any(a == -first for a, b in arcs)
        covered = (own & executed) if entered and not collision else set()
        visitor = ComplexityVisitor.from_ast(node)
        cc = visitor.functions[0].complexity
        ratio = len(covered) / len(own) if own else 0.0
        rows.append(
            dict(
                file=display_name or source_path.name,
                function=name,
                line=node.lineno,
                code_first_line=first,
                async_function=isinstance(node, ast.AsyncFunctionDef),
                decorator_lines=[d.lineno for d in node.decorator_list],
                cc=cc,
                body_lines=sorted(own),
                covered_lines=sorted(covered),
                missing_lines=sorted(own - covered),
                line_coverage=ratio,
                entry_arc_observed=entered,
                ambiguous_entry_collisions=collision,
                crap=cc * cc * (1 - ratio) ** 3 + cc,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage-json', type=Path, required=True)
    parser.add_argument('--coverage-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--source-snapshot',
        type=Path,
        help='Read source from a preserved repo-relative snapshot while keeping original coverage file identities',
    )
    args = parser.parse_args()
    repo = Path.cwd()
    report = json.loads(args.coverage_json.read_text(encoding='utf-8'))
    rows = []
    fingerprints = {}
    for file, data in report['files'].items():
        path = (repo / file).resolve()
        relative = path.relative_to(repo)
        content = args.source_snapshot / relative if args.source_snapshot else path
        raw = content.read_bytes()
        text = raw.decode('utf-8')
        fingerprints[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
        rows.extend(function_rows(path, args.coverage_data, data, text, relative.as_posix()))
    rows.sort(key=lambda row: row['crap'], reverse=True)
    totals = report['totals']
    result = dict(
        scope='Every .py file under src/plane_conditioned_segmentation; no omit or excluded lines; module/class code included in whole-source totals.',
        tools=dict(coverage=coverage.__version__, cc='radon6 ComplexityVisitor, assertions included'),
        line=dict(
            covered=totals['covered_lines'],
            total=totals['num_statements'],
            percent=100 * totals['covered_lines'] / totals['num_statements'],
        ),
        branch=dict(
            covered=totals['covered_branches'],
            total=totals['num_branches'],
            percent=100 * totals['covered_branches'] / totals['num_branches'],
        ),
        functions=len(rows),
        crap_over_30=sum(r['crap'] > 30 for r in rows),
        ambiguous_functions=[r['function'] for r in rows if r['ambiguous_entry_collisions']],
        source_sha256=fingerprints,
        rows=rows,
        method='CRAP = CC^2*(1-function-line-coverage)^3+CC. Executable body lines come from each compiled named code object plus anonymous comprehensions/lambdas owned by it, intersected with coverage statements. Nested named function bodies are separate; nested definition/decorator execution belongs to its parent. Def/decorator import execution alone never marks a function covered: a negative entry arc at co_firstlineno is required. Unexecuted functions use coverage 0. External decorator-lambda first-line collisions are reported and conservatively scored at 0. Async body requires actual execution, not merely constructing a coroutine. Branch coverage is reported separately, not used in CRAP. Semicolon/multiple expressions on one physical line remain line-level, not expression-level evidence. Module/class bodies and module lambdas are not CC function rows but remain in package totals.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ['rows', 'source_sha256', 'method']}, ensure_ascii=False
        )
    )
    print(
        json.dumps(
            [{k: r[k] for k in ['file', 'function', 'cc', 'line_coverage', 'crap']} for r in rows[:8]],
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
