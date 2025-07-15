"""
Python AST (Abstract Syntax Tree) loader for HERMES.

Extracts rich structural information from Python code to enhance embeddings
and provide better metadata for dimensional analysis.
"""

import ast
import sys
from typing import Dict, Any, List, Union, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging

from hermes.core.base import BaseLoader


logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a code symbol."""
    name: str
    type: str  # function, class, method, variable, import
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None  # For nested symbols
    complexity: int = 0  # Cyclomatic complexity for functions
    calls: Set[str] = field(default_factory=set)  # Functions this symbol calls
    imports: Set[str] = field(default_factory=set)  # What it imports/uses


class PythonASTLoader(BaseLoader):
    """
    Advanced Python code loader using AST analysis.
    
    Extracts:
    - Symbol table (functions, classes, methods, variables)
    - Import dependencies
    - Call graphs
    - Docstrings and comments
    - Complexity metrics
    - Semantic chunks based on code structure
    """
    
    def __init__(
        self,
        extract_docstrings: bool = True,
        extract_comments: bool = True,
        calculate_complexity: bool = True,
        chunk_by_symbols: bool = True,
        min_chunk_lines: int = 10,
        max_chunk_lines: int = 100,
    ):
        """
        Initialize Python AST loader.
        
        Args:
            extract_docstrings: Extract docstrings from functions/classes
            extract_comments: Extract inline comments
            calculate_complexity: Calculate cyclomatic complexity
            chunk_by_symbols: Chunk code by logical units (functions/classes)
            min_chunk_lines: Minimum lines per chunk
            max_chunk_lines: Maximum lines per chunk
        """
        self.extract_docstrings = extract_docstrings
        self.extract_comments = extract_comments
        self.calculate_complexity = calculate_complexity
        self.chunk_by_symbols = chunk_by_symbols
        self.min_chunk_lines = min_chunk_lines
        self.max_chunk_lines = max_chunk_lines
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load Python file and extract AST information.
        
        Returns:
            Dictionary with:
            - text: Full source code
            - ast: Parsed AST
            - symbols: Symbol table with all definitions
            - imports: Import statements and dependencies
            - call_graph: Function call relationships
            - chunks: Code chunks based on logical units
            - metadata: File-level metadata
            - complexity_metrics: Code complexity analysis
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Python file not found: {path}")
        
        # Read source code
        with open(path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        result = {
            "source_path": str(path),
            "text": source_code,
            "language": "python",
            "ast": None,
            "symbols": {},
            "imports": [],
            "call_graph": {},
            "chunks": [],
            "metadata": {
                "filename": path.name,
                "module_name": path.stem,
                "package": self._get_package_name(path),
                "lines_of_code": len(source_code.splitlines()),
            },
            "complexity_metrics": {},
            "content_type": "code"
        }
        
        try:
            # Parse AST
            tree = ast.parse(source_code, filename=str(path))
            result["ast"] = tree
            
            # Extract symbols and structure
            symbol_extractor = SymbolExtractor(
                source_code,
                extract_docstrings=self.extract_docstrings,
                calculate_complexity=self.calculate_complexity
            )
            symbol_extractor.visit(tree)
            
            result["symbols"] = symbol_extractor.symbols
            result["imports"] = symbol_extractor.imports
            result["call_graph"] = symbol_extractor.call_graph
            
            # Calculate metrics
            result["complexity_metrics"] = self._calculate_metrics(
                tree, symbol_extractor.symbols
            )
            
            # Create semantic chunks
            if self.chunk_by_symbols:
                result["chunks"] = self._create_semantic_chunks(
                    source_code, symbol_extractor.symbols
                )
            else:
                result["chunks"] = self._create_line_chunks(source_code)
            
            # Extract module-level docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                result["metadata"]["docstring"] = module_docstring
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {path}: {e}")
            result["metadata"]["syntax_error"] = str(e)
        
        return result
    
    def can_load(self, path: Union[str, Path]) -> bool:
        """Check if this loader can handle the file."""
        path = Path(path)
        return path.suffix.lower() in [".py", ".pyw"]
    
    def prepare_for_jina(self, ast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare AST data for Jina v4 embedding with code adapter.
        
        Enriches code with:
        - Symbol table context
        - Import context
        - Structural information
        - Semantic chunks
        
        Returns:
            Dictionary ready for Jina v4 processing
        """
        items = []
        item_types = []
        item_metadata = []
        
        # Process each semantic chunk
        for chunk in ast_data["chunks"]:
            # Enrich chunk with context
            enriched_text = self._enrich_chunk_with_context(
                chunk, ast_data["symbols"], ast_data["imports"]
            )
            
            items.append(enriched_text)
            item_types.append("code")  # Use Jina's code adapter
            item_metadata.append({
                "type": "code_chunk",
                "symbols": chunk.get("symbols", []),
                "complexity": chunk.get("complexity", 0),
                "imports": chunk.get("imports", []),
                "lines": f"{chunk['line_start']}-{chunk['line_end']}",
                "chunk_type": chunk.get("chunk_type", "mixed")
            })
        
        # Add symbol table as documentation
        symbol_doc = self._create_symbol_documentation(ast_data["symbols"])
        if symbol_doc:
            items.append(symbol_doc)
            item_types.append("text")  # Symbol docs as text
            item_metadata.append({
                "type": "symbol_table",
                "filename": ast_data["metadata"]["filename"]
            })
        
        return {
            "items": items,
            "item_types": item_types,
            "item_metadata": item_metadata,
            "document_metadata": ast_data["metadata"],
            "source_path": ast_data["source_path"],
            "complexity_metrics": ast_data["complexity_metrics"]
        }
    
    def _create_semantic_chunks(
        self, 
        source_code: str, 
        symbols: Dict[str, SymbolInfo]
    ) -> List[Dict[str, Any]]:
        """Create chunks based on code structure (functions, classes)."""
        lines = source_code.splitlines()
        chunks = []
        
        # Sort symbols by line number
        sorted_symbols = sorted(
            symbols.values(), 
            key=lambda s: s.line_start
        )
        
        # Group related symbols
        current_chunk = {
            "line_start": 1,
            "line_end": 1,
            "symbols": [],
            "text": "",
            "chunk_type": "module_header"
        }
        
        for symbol in sorted_symbols:
            # Check if symbol should start new chunk
            if (symbol.type in ["class", "function"] and 
                symbol.parent is None and
                current_chunk["symbols"]):
                
                # Finalize current chunk
                current_chunk["line_end"] = symbol.line_start - 1
                current_chunk["text"] = "\n".join(
                    lines[current_chunk["line_start"]-1:current_chunk["line_end"]]
                )
                chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = {
                    "line_start": symbol.line_start,
                    "line_end": symbol.line_end,
                    "symbols": [symbol.name],
                    "text": "",
                    "chunk_type": symbol.type
                }
            else:
                # Add to current chunk
                current_chunk["symbols"].append(symbol.name)
                current_chunk["line_end"] = max(
                    current_chunk["line_end"], 
                    symbol.line_end
                )
                if symbol.type in ["class", "function"]:
                    current_chunk["chunk_type"] = symbol.type
        
        # Finalize last chunk
        if current_chunk["symbols"] or current_chunk["line_end"] > current_chunk["line_start"]:
            current_chunk["text"] = "\n".join(
                lines[current_chunk["line_start"]-1:current_chunk["line_end"]]
            )
            chunks.append(current_chunk)
        
        # Add complexity metrics to chunks
        for chunk in chunks:
            chunk["complexity"] = sum(
                symbols.get(sym, SymbolInfo("", "", 0, 0)).complexity 
                for sym in chunk["symbols"]
            )
            
            # Extract imports used in chunk
            chunk["imports"] = []
            for sym_name in chunk["symbols"]:
                if sym_name in symbols:
                    chunk["imports"].extend(symbols[sym_name].imports)
            chunk["imports"] = list(set(chunk["imports"]))
        
        return chunks
    
    def _create_line_chunks(self, source_code: str) -> List[Dict[str, Any]]:
        """Create simple line-based chunks."""
        lines = source_code.splitlines()
        chunks = []
        
        for i in range(0, len(lines), self.max_chunk_lines):
            chunk_lines = lines[i:i + self.max_chunk_lines]
            chunks.append({
                "line_start": i + 1,
                "line_end": i + len(chunk_lines),
                "text": "\n".join(chunk_lines),
                "symbols": [],
                "chunk_type": "lines"
            })
        
        return chunks
    
    def _enrich_chunk_with_context(
        self, 
        chunk: Dict[str, Any],
        symbols: Dict[str, SymbolInfo],
        imports: List[Dict[str, Any]]
    ) -> str:
        """Add contextual information to code chunk."""
        enriched = []
        
        # Add imports context
        if chunk.get("imports"):
            enriched.append(f"# Uses: {', '.join(chunk['imports'])}")
        
        # Add symbol context
        if chunk.get("symbols"):
            symbol_lines = []
            for sym_name in chunk["symbols"]:
                if sym_name in symbols:
                    sym = symbols[sym_name]
                    if sym.docstring:
                        symbol_lines.append(f"# {sym_name}: {sym.docstring.splitlines()[0]}")
            if symbol_lines:
                enriched.extend(symbol_lines)
        
        # Add the actual code
        enriched.append(chunk["text"])
        
        return "\n".join(enriched)
    
    def _create_symbol_documentation(self, symbols: Dict[str, SymbolInfo]) -> str:
        """Create a documentation summary of all symbols."""
        if not symbols:
            return ""
        
        doc_lines = ["# Symbol Table", ""]
        
        # Group by type
        by_type = {}
        for name, info in symbols.items():
            by_type.setdefault(info.type, []).append((name, info))
        
        # Document each type
        for sym_type in ["class", "function", "method", "variable"]:
            if sym_type in by_type:
                doc_lines.append(f"## {sym_type.title()}s:")
                for name, info in sorted(by_type[sym_type]):
                    line = f"- {name}"
                    if info.parent:
                        line += f" (in {info.parent})"
                    if info.docstring:
                        line += f": {info.docstring.splitlines()[0]}"
                    doc_lines.append(line)
                doc_lines.append("")
        
        return "\n".join(doc_lines)
    
    def _calculate_metrics(
        self, 
        tree: ast.AST, 
        symbols: Dict[str, SymbolInfo]
    ) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        metrics = {
            "total_lines": len(tree.body),
            "num_functions": sum(1 for s in symbols.values() if s.type == "function"),
            "num_classes": sum(1 for s in symbols.values() if s.type == "class"),
            "num_methods": sum(1 for s in symbols.values() if s.type == "method"),
            "avg_complexity": 0,
            "max_complexity": 0,
        }
        
        complexities = [s.complexity for s in symbols.values() if s.complexity > 0]
        if complexities:
            metrics["avg_complexity"] = sum(complexities) / len(complexities)
            metrics["max_complexity"] = max(complexities)
        
        return metrics
    
    def _get_package_name(self, path: Path) -> str:
        """Determine package name from file path."""
        parts = []
        current = path.parent
        
        while current != current.parent:
            if (current / "__init__.py").exists():
                parts.append(current.name)
                current = current.parent
            else:
                break
        
        return ".".join(reversed(parts)) if parts else ""


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor to extract symbols and relationships."""
    
    def __init__(self, source_code: str, extract_docstrings: bool = True, 
                 calculate_complexity: bool = True):
        self.source_code = source_code
        self.extract_docstrings = extract_docstrings
        self.calculate_complexity = calculate_complexity
        self.symbols = {}
        self.imports = []
        self.call_graph = {}
        self.current_class = None
        self.current_function = None
        self.complexity_stack = []
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definitions."""
        symbol = SymbolInfo(
            name=node.name,
            type="class",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node) if self.extract_docstrings else None,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            parent=self.current_class
        )
        
        self.symbols[node.name] = symbol
        
        # Visit class body
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function definitions."""
        self._visit_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Extract async function definitions."""
        self._visit_function(node, is_async=True)
    
    def _visit_function(self, node, is_async=False):
        """Common function processing."""
        func_type = "method" if self.current_class else "function"
        symbol = SymbolInfo(
            name=node.name,
            type=func_type,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node) if self.extract_docstrings else None,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            parent=self.current_class
        )
        
        # Calculate complexity
        if self.calculate_complexity:
            symbol.complexity = self._calculate_complexity(node)
        
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.symbols[full_name] = symbol
        
        # Visit function body
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Import(self, node: ast.Import):
        """Extract import statements."""
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "import"
            })
            
            if self.current_function:
                self.symbols[self.current_function].imports.add(alias.name)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Extract from imports."""
        module = node.module or ""
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports.append({
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "from_import"
            })
            
            if self.current_function:
                self.symbols[self.current_function].imports.add(full_name)
    
    def visit_Call(self, node: ast.Call):
        """Track function calls for call graph."""
        if self.current_function and isinstance(node.func, ast.Name):
            called_func = node.func.id
            self.symbols[self.current_function].calls.add(called_func)
            
            # Update call graph
            if self.current_function not in self.call_graph:
                self.call_graph[self.current_function] = set()
            self.call_graph[self.current_function].add(called_func)
        
        self.generic_visit(node)
    
    def _get_decorator_name(self, decorator):
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        return "unknown"
    
    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity