from typing import List, Set

class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag 

class DependencyEdge:
    def __init__(self, source: Token, target: Token, label:str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label
        pass


class ParseState:
    def __init__(self, stack: List[Token], parse_buffer: List[Token], dependencies: List[DependencyEdge], includeDep=False, pad="[PAD]"):
        self.stack = stack # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies 
        self.pad = pad
        self.includeDep = includeDep
        pass

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )

    def getLeftmostChild(self, token: Token):
        leftmostPos = None
        leftmostTok = None
        leftmostDep = None
        for i in range(len(self.dependencies)):
            if self.dependencies[i].source == token:
                if not leftmostPos or self.dependencies[i].target.idx < leftmostPos:
                    leftmostPos = self.dependencies[i].target.idx
                    leftmostTok = self.dependencies[i].target
                    leftmostDep = self.dependencies[i].label
        return leftmostTok, leftmostDep

    def getRightmostChild(self, token: Token):
        rightmostPos = None
        rightmostTok = None
        rightmostDep = None
        for i in range(len(self.dependencies)):
            if self.dependencies[i].source == token:
                if not rightmostPos or self.dependencies[i].target.idx > rightmostPos:
                    rightmostPos = self.dependencies[i].target.idx
                    rightmostTok = self.dependencies[i].target
                    rightmostDep = self.dependencies[i].label
        return rightmostTok, rightmostDep

    def getPad(self):
        return self.pad

    def getRepr(self, context=2):
        pad = self.getPad()
        w, p = [pad]*2*context, ["NULL"]*2*context
        if self.includeDep:
            l = ["NULL"]*2*context
            wApp, pApp = [pad]*2*context, ["NULL"]*2*context
        stackL, buffL = min(context, len(self.stack)), min(context, len(self.parse_buffer))
        if stackL:
            for i in range(len(self.stack)-stackL, len(self.stack)):
                w[context + (i-len(self.stack))] = self.stack[i].word
                p[context + (i-len(self.stack))] = self.stack[i].pos
                if self.includeDep:
                    leftmostTok, leftmostDep = self.getLeftmostChild(self.stack[i])
                    if leftmostTok:
                        wApp[2*(i-len(self.stack))] = leftmostTok.word
                        pApp[2*(i-len(self.stack))] = leftmostTok.pos
                        l[2*(i-len(self.stack))] = leftmostDep
                    rightmostTok, rightmostDep = self.getRightmostChild(self.stack[i])
                    if rightmostTok:
                        wApp[2*(i-len(self.stack))+1] = rightmostTok.word
                        pApp[2*(i-len(self.stack))+1] = rightmostTok.pos
                        l[2*(i-len(self.stack))+1] = rightmostDep
        if buffL:
            for i in range(buffL):
                w[context+i] = self.parse_buffer[i].word
                p[context+i] = self.parse_buffer[i].pos

        out = {}
        if self.includeDep:
            w.extend(wApp)
            p.extend(pApp)
            out["l"] = l
        out["w"] = w
        out["p"] = p

        return out
    
def isActionValid(state: ParseState, action: str, error: str="isActionValid"):
    if action == "SHIFT":
        return (len(state.parse_buffer)!=0)
    elif action.startswith("REDUCE_L"):
        if action == "REDUCE_L_root":
            return (len(state.stack)>=2) and state.stack[-1].idx==-1 and (len(state.parse_buffer)==0)
        return (len(state.stack)>=2) and state.stack[-2].idx!=-1 and state.stack[-1].idx!=-1
    elif action.startswith("REDUCE_R"):
        if action == "REDUCE_R_root":
            return (len(state.stack)>=2) and state.stack[-2].idx==-1 and (len(state.parse_buffer)==0)
        return (len(state.stack)>=2) and state.stack[-2].idx!=-1 and state.stack[-1].idx!=-1
    else: 
        raise RuntimeError("[{}] Unrecognized action: {}".format(error, action))

def shift(state: ParseState) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    if len(state.parse_buffer) == 0:
        raise RuntimeError("[shift] Not enough elements in the parse_buffer to perform a shift operation!")

    state.stack.append(state.parse_buffer.popleft())


def left_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.

    if len(state.stack) < 2:
        raise RuntimeError("[left_arc] Not enough elements in the stack to perform an arc operation!")
    
    e1, e2 = state.stack.pop(), state.stack.pop()
    state.add_dependency(e1, e2, label)
    state.stack.append(e1)


def right_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.

    if len(state.stack) < 2:
        raise RuntimeError("[right_arc] Not enough elements in the stack to perform an arc operation!")

    e1, e2 = state.stack.pop(), state.stack.pop()
    state.add_dependency(e2, e1, label)
    state.stack.append(e2)



def is_final_state(state: ParseState, cwindow: int) -> bool:
    # TODO: Implemement this
    return ((len(state.parse_buffer)==0) and len(state.stack)==1 and state.stack[0].idx==-1)
