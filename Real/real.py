from tokens import *
from strings_with_arrows import *

# Token idetifier
class Token:
    def __init__(self, type_, value=None, ps=None, pe=None):
        self.type = type_
        self.value = value
        if ps:
            self.ps = ps.copy()
            self.pe = ps.copy()
            self.pe.advance()
        if pe:
            self.pe = pe

    def __repr__(self) -> str:
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


# Lexer
class Lexer:
    def __init__(self, fn, txt):
        self.txt = txt
        self.pos = Position(-1,0,-1, fn, txt)
        self.cchar = None
        self.advance()

    # Change the position
    def advance(self):
        self.pos.advance(self.cchar)
        self.cchar = self.txt[self.pos.idx] if self.pos.idx < len(self.txt) else None
    
    # Token maker
    def tokenize(self):
        tokens = []
        while self.cchar is not None:
            if self.cchar in ' \t':
                self.advance()
            elif self.cchar in DIGITS:
                tokens.append(self.numberize())
            elif self.cchar == '+':
                tokens.append(Token(TT_PLUS, ps=self.pos))
                self.advance()
            elif self.cchar == '-':
                tokens.append(Token(TT_MINUS, ps=self.pos))
                self.advance()
            elif self.cchar == '*':
                tokens.append(Token(TT_MUL, ps=self.pos))
                self.advance()
            elif self.cchar == '/':
                tokens.append(Token(TT_DIV, ps=self.pos))
                self.advance()
            elif self.cchar == '(':
                tokens.append(Token(TT_LPAREN, ps=self.pos))
                self.advance()
            elif self.cchar == ')':
                tokens.append(Token(TT_RPAREN, ps=self.pos))
                self.advance()
            else:
                ps = self.pos.copy()
                char = self.cchar
                self.advance()
                return [], IllegalCharError(ps, self.pos, "'" + char + "'")
        
        tokens.append(Token(TT_EOF, ps=self.pos))
        return tokens, None

    # Number maker
    def numberize(self):
        nstr = ''
        dcount  = 0
        ps = self.pos.copy()
        while self.cchar is not None and self.cchar in DIGITS + '.':
            if self.cchar == '.':
                if dcount == 1:
                    break
                dcount += 1
                nstr += '.'
            else:
                nstr += self.cchar
            self.advance()
        if dcount == 0:
            return Token(TT_INT, int(nstr), ps=ps, pe=self.pos)
        else:
            return Token(TT_FLOAT, float(nstr), ps=ps, pe=self.pos)

# Position
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    
    # advance
    def advance(self, cchar=None):
        self.idx += 1
        self.col += 1
        if cchar == '\n':
            self.ln += 1
            self.col = 0
        return self
    
    # copy the position
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

# ERRORS
class Error:
    def __init__(self, ps, pe, ename, edetails):
        self.ename = ename
        self.edetails = edetails
        self.ps = ps
        self.pe = pe
    
    def as_string(self):
            result  = f'{self.ename}: {self.edetails}\n'
            result += f'File {self.ps.fn}, line {self.ps.ln + 1}'
            result += '\n' + string_with_arrows(self.ps.ftxt, self.ps, self.pe)
            return result

class IllegalCharError(Error):
    def __init__(self, ps, pe, edetails,):
        super().__init__(ps, pe, 'ILLEGAL CHARACTER', edetails)

class InvalidSynthaxError(Error):
    def __init__(self, ps, pe, edetails,):
        super().__init__(ps, pe, 'INVALID SYNTHAX', edetails)

class RTError(Error):
    def __init__(self, ps, pe, edetails, ctx):
        super().__init__(ps, pe, 'RUNTIME ERROR', edetails)
        self.ctx = ctx

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.ename}: {self.edetails}\n'
        result += '\n' + string_with_arrows(self.ps.ftxt, self.ps, self.pe)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.ps
        ctx = self.ctx
        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.dname}\n' + result
            pos = pos
            ctx = ctx
        return 'Traceback (most recent call last):\n' + result


# Nodes
class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.ps = self.tok.ps
        self.pe = self.tok.pe
    
    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, lnode, op_tok, rnode):
        self.lnode = lnode
        self.optok = op_tok
        self.rnode = rnode
        self.ps = self.lnode.ps
        self.pe = self.rnode.pe
    
    def __repr__(self):
        return f'({self.lnode}, {self.optok}, {self.rnode})'

class UnaryOpNode:
    def __init__(self, optok, node):
        self.optok = optok
        self.node = node
        self.ps = self.optok.ps
        self.pe = self.node.pe

    def __repr__(self):
        return f'({self.optok}, {self.node})'    

# PARSE RESULT
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
    
    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        
        return res
    
    # CHECK IF IT IS A SUCCESS
    def success(self, node):
        self.node = node
        return self

    # CHECK IF THERE IS AN ERROR
    def failure(self, error):
        self.error = error
        return self

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokidx = -1
        self.advance()
    
    # Advance
    def advance(self):
        self.tokidx += 1
        if self.tokidx < len(self.tokens):
            self.ctok = self.tokens[self.tokidx]
        return self.ctok
    
    # Parse method
    def parse(self):
        res = self.expr()
        if not res.error and self.ctok.type != TT_EOF:
            return res.failure(
                InvalidSynthaxError(
                    ps = self.ctok.ps,
                    pe = self.ctok.pe,
                    edetails = "Expected '+', '-', '*' or '/'"
                )
            )
        return res

    #################################################
    def factor(self):
        res = ParseResult()
        tok = self.ctok
        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        # Here, we have to check if the current token type is a LPAREN
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.ctok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSynthaxError(
                        ps = self.ctok.ps,
                        pe = self.ctok.pe,
                        edetails = "Expected ')'"
                    )
                )

        return res.failure(
            InvalidSynthaxError(
                tok.ps,
                tok.pe,
                'Expected int or float'
            )
        )

    def term(self):
        return self.binop(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.binop(self.term, (TT_PLUS, TT_MINUS))

    def binop(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res
        while self.ctok.type in ops:
            optok = self.ctok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = BinOpNode(left, optok, right)
        return res.success(left)


# RUNTIME RESULT
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

# VALUES
class Number:
    def __init__(self,value):
        self.value = value
        self.spos()
        self.sctx()
    
    def spos(self, ps=None, pe=None):
        self.ps = ps
        self.pe = pe
        return self
    
    def sctx(self, ctx = None):
        self.ctx = ctx
        return self
    
    def addto(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).sctx(self.ctx), None
    
    def subto(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).sctx(self.ctx), None

    def multo(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).sctx(self.ctx), None

    def divto(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.ps,
                    other.pe,
                    'Division by zero is illegal',
                    self.ctx
                )
            return Number(self.value / other.value).sctx(self.ctx), None
    
    def __repr__(self):
        return str(self.value)

# CONTEXT
class Context:
    def __init__(self, dname, parent=None, pep=None):
        self.dname = dname
        self.parent = parent
        self.pep = pep
        self.symbol_table = None
    
    def find(self, name):
        # First, check if the name is present in the local scope
        if self.symbol_table.contains(name):
            return self.symbol_table.get(name)
        
        # If not found, check in the parent
        if self.parent:
            return self.parent.find(name)
        
        return None

# INTERPRETER
class Interpreter:
    def visit(self, node, context):
        mname = f'visit_{type(node).__name__}'
        method = getattr(self, mname, self.nvm)
        return method(node, context)

    def nvm(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined.')
    
    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(
                node.tok.value
            ).sctx(context).spos(
                    node.tok.ps, node.tok.pe
                )
        )
    
    def visit_BinOpNode(self, node, context):
        res = RTResult()
        l = res.register(self.visit(node.lnode, context))
        if res.error: return res
        r = res.register(self.visit(node.rnode, context))
        if res.error: return res
        if node.optok.type == TT_PLUS:
            result, error = l.addto(r)
        elif node.optok.type == TT_MINUS:
            result, error = l.subto(r)
        elif node.optok.type == TT_MUL:
            result, error = l.multo(r)
        elif node.optok.type == TT_DIV:
            result, error = l.divto(r)
        
        if error: return res.failure(error)
        else:
            return res.success(result.spos(node.ps, node.pe))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        num = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.optok.type == TT_MINUS:
            num, error = num.multo(Number(-1))
        
        if res.error: return res.failure(error)
        else:
            return res.success(num.spos(node.ps, node.pe))


# Finally, run the interpreter:
def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, errors = lexer.tokenize()
    if errors: return None, errors

    # GENERATE AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # RUN INTERPRETER
    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)
    return result.value, result.error