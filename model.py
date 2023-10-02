import argparse
import logging
import numpy as np
from pathlib import Path
from os.path import exists
import os
import glob
import torch
import pandas as pd
import transformers
import math
from tqdm import tqdm
from scripts.state import *
from collections import deque
from torchtext.vocab import GloVe
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-train",
    help="Path to train file",
    # default="./data/train.txt"
)

parser.add_argument(
    "-val",
    help="Path to validation file",
    # default="./data/dev.txt"
)

parser.add_argument(
    "-test",
    help="Path to test file",
    # default="./data/test.txt"
)

parser.add_argument(
    "-predict",
    help="Path to file containing instances to make predictions on",
    # default="./data/hidden.txt"
)

parser.add_argument(
    "-pos",
    help="Path to pos set file",
    default="./data/pos_set.txt"
)

parser.add_argument(
    "-tag",
    help="Path to tag set file",
    default="./data/tagset.txt"
)

parser.add_argument(
    "-out",
    help="Path to directory where outputs should be saved",
    default="./out/"
)

parser.add_argument(
    "-gloveName",
    choices=["6B", "42B", "840B"],
    help="Name of Glove embedding",
    default="6B"
)

parser.add_argument(
    "-gloveDim",
    type=int,
    choices=[50, 300],
    help="Size of Glove embedding",
    default=50
)

parser.add_argument(
    "-embedPos",
    type=int,
    help="Size of POS tag embedding",
    default=50
)

parser.add_argument(
    "-embedLabel",
    type=int,
    help="Size of Label tag embedding",
    default=50
)

parser.add_argument(
    "-hiddenDim",
    type=int,
    help="Size of hidden dimension",
    default=200
)

parser.add_argument(
    "-c",
    "--context",
    type=int,
    help="Size of context window",
    default=2
)

parser.add_argument(
    "-strategy",
    choices=["mean", "concat"],
    help="Strategy to use for combining word embeddings before being fed to model",
    default="mean"
)

parser.add_argument(
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=20
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=256
)

parser.add_argument(
    "-learningRate",
    type=float,
    nargs="+",
    help="Learning rate(s) for optimizer",
    default=[0.01, 0.001, 0.0001]
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0.01
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=-1
)

parser.add_argument(
    "-load",
    type=str,
    help="[Optional] Path to saved PyTorch model to load"
)

parser.add_argument(
    "-includeDep",
    action="store_true",
    help="Boolean flag to add leftmost and rightmost words of context words to word repr along with their dependency labels"
)

parser.add_argument(
    "-dropout",
    type=float,
    help="Dropout probability",
    default=0.5
)
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False, error="checkIfExists"): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"[{error}] {path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"[{error}] {path} is not a file!")   
    return path
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None, error="checkIfExists"): 
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[{error}] {fileName} does not have expected file extension {fileExtension}!")
    return checkIfExists(fileName, isDir=False, createIfNotExists=False, error=error)
#---------------------------------------------------------------------------
def readFile(fileName, error="readFile"):
    data = []
    fileExt = fileName.split(".")[-1]
    if fileExt == "txt":
        with open(fileName, "r") as f: 
            data = list(f.readlines())
            data = list(map(str.strip, data))
    else: 
        raise ValueError(f"[{error}] Unsupported file type: {fileExt}")
    return data
#---------------------------------------------------------------------------
def writeFile(data: List[str], fileName: str, error: str="writeFile"):
    fileExt = fileName.split(".")[-1]
    if fileExt == "txt":
            with open(fileName, "w") as f: 
                for d in data:
                    f.write(d)
                    f.write("\n")
    else: 
        raise ValueError(f"[{error}] Unsupported file type: {fileExt}")
    return data
#---------------------------------------------------------------------------
def extractData(data, error="extractData"):
    extrData = []
    for d in data:
        pass
        assert type(d)==str
        if len(d) == 0:
            continue
        splitD = d.split("|||")
        curData = {}
        curData["wordSeq"] = splitD[0].strip()
        curData["words"] = splitD[0].strip().split(" ")
        if len(splitD)>1:
            curData["posSeq"] = splitD[1].strip()
            curData["pos"] = splitD[1].strip().split(" ")
        if len(splitD)>2:
            curData["actionSeq"] = splitD[2].strip()
            curData["actions"] = splitD[2].strip().split(" ")
        extrData.append(curData)
    if len(extrData) == 0:
        raise RuntimeWarning(f"[{error}] Could not process data!")
    return extrData
#---------------------------------------------------------------------------
def takeAction(ps: ParseState, action: str, error: str="takeAction"):
    assert isActionValid(ps, action, error)
    if action == "SHIFT":
        shift(ps)
    elif action.startswith("REDUCE_L"):
        left_arc(ps, action.split("_")[-1])
    elif action.startswith("REDUCE_R"):
        right_arc(ps, action.split("_")[-1])
    else: 
        raise RuntimeError("[{}] Unrecognized action: {}".format(error, action))
#---------------------------------------------------------------------------
def processData(extrData: List[dict], context: int=2, includeDep=False, error: str="processData"):
    proData = []
    proActions = []
    for data in extrData:
        stack = [Token(-1, "root", "NULL")] 
        parser_buffer = deque([])
        for i, word in enumerate(data["words"]):
            parser_buffer.append(Token(i, word, data["pos"][i]))
        ps = ParseState(
            stack=stack,
            parse_buffer=parser_buffer,
            dependencies=[],
            includeDep=includeDep
        )
        for action in data["actions"]:
            takeAction(ps, action, error="processData")
            proData.append(ps.getRepr(context))
            proActions.append(action)
        takeAction(ps, "REDUCE_R_root", error="processData")
        if not is_final_state(ps, context):
            raise RuntimeError("[{}] Final state not reached after all actions: {}".format(error, ps.getRepr(context)))
        proData.append(ps.getRepr(context))
        proActions.append("REDUCE_R_root")
    return proData, proActions
#---------------------------------------------------------------------------
class Actor(torch.nn.Module):
    def __init__(self, context, gloveDim, posDim, hiddenSize, posToInd, labelToInd, strategy="mean", device="cpu", includeDep=False, depToInd=None, labelDim=50, dropout=0.5):
        super(Actor, self).__init__()
        self.context = context
        self.gloveDim = gloveDim
        self.posDim = posDim
        self.hiddenSize = hiddenSize
        self.posToInd = posToInd
        self.labelToInd = labelToInd
        self.strategy = strategy
        self.device = device
        self.includeDep = includeDep
        self.depToInd = depToInd
        self.labelDim = labelDim

        if self.strategy == "mean":
            self.embedDim = gloveDim
            self.embedPosDim = posDim
            if includeDep:
                self.embedLabelDim = labelDim
        elif self.strategy == "concat":
            self.embedDim = gloveDim*2*context
            self.embedPosDim = posDim*2*context
            if includeDep:
                self.embedDim += gloveDim*2*context
                self.embedPosDim += posDim*2*context
                self.embedLabelDim = labelDim*2*context
        else: 
            raise ValueError("[Actor::__init__] Unrecognized strategy: {}".format(self.strategy))

        self.wordLayer = torch.nn.Linear(
            in_features=self.embedDim, 
            out_features=hiddenSize, 
        )
        
        # self.wordBN = torch.nn.BatchNorm1d(self.hiddenSize)

        self.posEmbed = torch.nn.Embedding(
            num_embeddings=len(posToInd), 
            embedding_dim=posDim,
            max_norm=1,
        )

        self.posLayer = torch.nn.Linear(
            in_features=self.embedPosDim, 
            out_features=hiddenSize, 
        )

        # self.posBN = torch.nn.BatchNorm1d(self.hiddenSize)

        if includeDep:
            self.labelEmbed = torch.nn.Embedding(
                num_embeddings=len(depToInd)+1, 
                embedding_dim=labelDim,
                max_norm=1,
            )
            self.labelLayer = torch.nn.Linear(
                in_features=self.embedLabelDim, 
                out_features=hiddenSize, 
            )

        self.dropout = torch.nn.Dropout(p=dropout)

        self.classifier = torch.nn.Linear(
            in_features=hiddenSize, 
            out_features=len(labelToInd), 
        )
        self.softmax = torch.nn.Softmax(dim=-1)

        self.to(device)

    def forward(self, w, p, l=None):
        w = w.to(device=self.device, dtype=self.wordLayer.weight.dtype)
        p = p.to(device=self.device)
        pEmbed = self.posEmbed(p)
        if self.includeDep:
            if l==None: 
                raise RuntimeError("[Actor::forward] l must be provided alond with w and p when includeDep is set to True!")
            l = l.to(device=self.device)
            lEmbed = self.labelEmbed(l)
        if self.strategy == "mean":
            w = torch.mean(w, dim=-2)
            pEmbed = torch.mean(pEmbed, dim=-2)
            if self.includeDep:
                lEmbed = torch.mean(lEmbed, dim=-2)
        elif self.strategy == "concat":
            w = torch.flatten(w, start_dim=-2, end_dim=-1)
            pEmbed = torch.flatten(pEmbed, start_dim=-2, end_dim=-1)
            if self.includeDep:
                lEmbed = torch.flatten(lEmbed, start_dim=-2, end_dim=-1)
        else: 
            raise ValueError("[Actor::forward] Unrecognized strategy: {}".format(self.strategy))
        if self.includeDep:
            h = self.wordLayer(w) + self.posLayer(pEmbed) + self.labelLayer(lEmbed)
        else: 
            # h = torch.nn.ReLU()(self.wordBN(self.wordLayer(w)) + self.posBN(self.posLayer(pEmbed)))
            h = self.wordLayer(w) + self.posLayer(pEmbed)
        h = torch.nn.ReLU()(self.dropout(h))
        logits = self.classifier(h)
        probs = self.softmax(logits)
        return probs, logits

    def to(self, device):
        self.device = device 
        self = super(Actor, self).to(device)
        return self
    
    def getContext(self):
        return self.context
    
    def getModelParams(self):
        return {
            "context": self.context,
            "gloveDim": self.gloveDim,
            "posDim": self.posDim,
            "hiddenSize": self.hiddenSize,
            "posToInd": self.posToInd,
            "labelToInd": self.labelToInd,
            "strategy": self.strategy,
            "device": self.device,
            "includeDep": self.includeDep,
            "labelDim": self.labelDim,
        }
#---------------------------------------------------------------------------
class ActorDataset:
    def __init__(self, labelToInd, posToInd, context, gloveName="6B", gloveDim=50, includeDep=False, depToInd=None):
        self.data = []
        self.labels = None
        self.labelToInd = labelToInd
        self.posToInd = posToInd
        self._posToInd = lambda pos: posToInd[pos]
        self.context = context
        self.gloveName = gloveName
        self.gloveDim = gloveDim
        self.includeDep = includeDep
        self.depToInd = depToInd
        self._depToInd = lambda dep: depToInd[dep]
        self.gloveVecs = GloVe(
            name=gloveName,
            dim=gloveDim
        )
    
    def addData(self, data, labels=None):
        if labels:
            assert len(data) == len(labels)
        if self.labels and labels:
            self.data.extend(data)
            self.labels.extend(labels)
        else:
            if labels and (not self.labels) and len(self.data):
                raise RuntimeError("[ActorDataset::addData] Cannot add data with labels to data without labels!")
            self.data = data
            self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        if len(self.data) == 0:
            raise RuntimeError("[ActorDataset::__getitem__] No data added!")
        curInstance = {}
        curInstance["w"] = self.gloveVecs.get_vecs_by_tokens(self.data[item]["w"]).tolist()
        curInstance["p"] = list(map(self._posToInd, self.data[item]["p"]))
        if self.includeDep:
            curInstance["l"] = list(map(self._depToInd, self.data[item]["l"]))
        if self.labels:
            curInstance["labels"] = self.labelToInd[self.labels[item]]

        curInstance["w"] = torch.tensor(curInstance["w"])
        curInstance["p"] = torch.tensor(curInstance["p"])
        if self.includeDep:
            curInstance["l"] = torch.tensor(curInstance["l"])
        if self.labels:
            curInstance["labels"] = torch.tensor(curInstance["labels"])

        if self.labels:
            if self.includeDep:
                return curInstance["w"], curInstance["p"], curInstance["l"], curInstance["labels"]
            else:
                return curInstance["w"], curInstance["p"], curInstance["labels"]
        else:
            if self.includeDep:
                return curInstance["w"], curInstance["p"], curInstance["l"], None
            else:
                return curInstance["w"], curInstance["p"], None
#---------------------------------------------------------------------------
class collateBatch:
    def __init__(self, includeDep=False):
        self.includeDep = includeDep
    def __call__(self, batch):
        if self.includeDep:
            Ws, Ps, Ls, labels = zip(*batch)
            Ws = torch.stack(Ws)
            Ps = torch.stack(Ps)
            Ls = torch.stack(Ls)
            labels = torch.stack(labels)
            return Ws, Ps, Ls, labels
        else:
            Ws, Ps, labels = zip(*batch)
            Ws = torch.stack(Ws)
            Ps = torch.stack(Ps)
            labels = torch.stack(labels)
            return Ws, Ps, labels
#---------------------------------------------------------------------------
def createDataLoader(data, labels, labelToInd, posToInd, context, gloveName, gloveDim, batchSize, includeDep=False, depToInd=None):
    ds = ActorDataset(
        labelToInd,
        posToInd,
        context,
        gloveName=gloveName,
        gloveDim=gloveDim,
        includeDep=includeDep,
        depToInd=depToInd
    ) 

    ds.addData(
        data=data,
        labels=labels,
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        # shuffle=False,
        collate_fn=collateBatch(includeDep),
    )
#---------------------------------------------------------------------------
def trainModel(model: Actor, dataLoader, lossFunction, optimizer, device, scheduler=None, maxSteps=-1, logSteps=1000, includeDep=False, dataDesc="Train batch"):
    model.to(device)
    model.train()

    losses = []
    corrPreds = 0
    numExamples = 0
    numBatch = 0
    numSteps = 0
    for d in tqdm(dataLoader, desc=dataDesc):
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        if includeDep:
            Ws, Ps, Ls, labels = d
        else:
            Ws, Ps, labels = d
        numBatch += 1
        numExamples += len(Ws)
        if includeDep:
            outputs, _ = model(Ws, Ps, Ls)
        else:
            outputs, _ = model(Ws, Ps)

        labels = labels.to(device)
        
        _, preds = torch.max(outputs, dim=-1)

        loss = lossFunction(outputs, labels)

        if numSteps%logSteps == 0:
            logging.info(f"\nBatch: {numBatch}/{len(dataLoader)}, Loss: {loss.item()}")

        corrPreds += torch.sum(preds == labels)
        losses.append(loss.item())
        #Backwardpropagate the losses
        loss.backward()
        #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
        numSteps += 1
        if maxSteps and numSteps >= maxSteps:
            break
    if scheduler:
        scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def evalModel(model, lossFunction, dataLoader, device="cpu", dataDesc="Test batch", includeDep=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        losses = []
        corrPreds = 0
        numExamples = 0
        numBatch = 0
        numSteps = 0
        for d in tqdm(dataLoader, desc=dataDesc):
            if includeDep:
                Ws, Ps, Ls, labels = d
            else:
                Ws, Ps, labels = d
            numBatch += 1
            numExamples += len(Ws)
            if includeDep:
                outputs, _ = model(Ws, Ps, Ls)
            else:
                outputs, _ = model(Ws, Ps)
            labels = labels.to(device)
            _, preds = torch.max(outputs, dim=-1)

            loss = lossFunction(outputs, labels)

            corrPreds += torch.sum(preds == labels)
            losses.append(loss.item())
            numSteps += 1
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def testModel(model: Actor, data: list, indToLabel: dict, datasetInst: ActorDataset, device: str="cpu", includeDep=False, dataDesc: str="Test data"):
    model.to(device)
    model.eval()
    context = model.getContext()
    _indToLabel = lambda x: indToLabel[x]
    words = []
    labels = []
    preds = []
    with torch.no_grad():
        for instance in tqdm(data, desc=dataDesc):
            words.append(instance["words"])
            if "actions" in instance.keys():
                labels.append(instance["actions"])
            predActions = []
            stack = [Token(-1, "root", "NULL")] 
            parser_buffer = deque([])
            for i, word in enumerate(instance["words"]):
                parser_buffer.append(Token(i, word, instance["pos"][i]))
            ps = ParseState(
                stack=stack,
                parse_buffer=parser_buffer,
                dependencies=[],
                includeDep=includeDep,
            )
            while True:
                repr = ps.getRepr(context)
                datasetInst.addData([repr])
                if includeDep:
                    w, p, l, _ = datasetInst.__getitem__(-1)
                    _, nextActionLog = model(w, p, l)
                else:
                    w, p, _ = datasetInst.__getitem__(-1)
                    _, nextActionLog = model(w, p)
                while(1):
                    if (nextActionLog.max() == -torch.inf).item():
                        raise RuntimeError("[testModel] Could not find a feasible action in state: {}".format(ps.getRepr(context)))
                    nextActionInd = torch.nn.Softmax(dim=-1)(nextActionLog).argmax().item()
                    nextAction = _indToLabel(nextActionInd)
                    try:
                        takeAction(ps, nextAction, error="testModel")
                        predActions.append(nextAction)
                        break
                    except: #If invalid action is predicted
                        nextActionLog[nextActionInd] = -torch.inf
                if is_final_state(ps, context):
                    preds.append(predActions[:-1])
                    break
    if len(labels) == 0:
        labels = None
    return words, labels, preds
#---------------------------------------------------------------------------
def get_deps(words_lists, actions, cwindow):
    """ Computes all the dependencies set for all the sentences according to 
    actions provided
    Inputs
    -----------
    words_lists: List[List[str]].  This is a list of lists. Each inner list is a list of words in a sentence,
    actions: List[List[str]]. This is a list of lists where each inner list is the sequence of actions
                Note that the elements should be valid actions as in `tagset.txt`
    cwindow: int. Context window. Default=2
    """
    all_deps = []   # List of List of dependencies
    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        stack = [Token(-1, "root", "NULL")] 
        parser_buff = deque([])
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos="NULL"))
        # parser_buff.extend([Token(idx=ix+i+1, word="[PAD]",pos="NULL") for i in range(cwindow)])
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        # Iterate over the actions and do the necessary state changes
        for action in actions[w_ix]:
            if action == "SHIFT":
                shift(state)
            elif action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
            else:
                right_arc(state, action[9:])
        right_arc(state, "root")    # Add te root dependency for the remaining element on stack
        assert is_final_state(state,cwindow)    # Check to see that the parse is complete
        all_deps.append(state.dependencies.copy())  # Copy over the dependenices found
    return all_deps
#---------------------------------------------------------------------------
def compute_metrics(words_lists, gold_actions, pred_actions, cwindow=2):
    """ Computes the UAS and LAS metrics given list of words, gold and predicted actions.
    Inputs
    -------
    word_lists: List[List[str]]. This is a list of lists. Each inner list is a list of words in a sentence,
    gold_action: List[List[str]]. This is a list of lists where each inner list is the sequence of gold actions
                Note that the elements should be valid actions as in `tagset.txt`
    pred_action: List[List[str]]. This is a list of lists where each inner list is the sequence of predicted actions
                Note that the elements should be valid actions as in `tagset.txt`
 
    Outputs
    -------
    uas: int. The Unlabeled Attachment Score
    las: int. The Lableled Attachment Score
    """
    lab_match = 0  # Counter for computing correct head assignment and dep label
    unlab_match = 0 # Counter for computing correct head assignments
    total = 0       # Total tokens

    # Get all the dependencies for all the sentences
    gold_deps = get_deps(words_lists, gold_actions, cwindow)    # Dep according to gold actions
    pred_deps = get_deps(words_lists, pred_actions, cwindow)    # Dep according to predicted actions

    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Iterate over words in a sentence
        for ix, word in enumerate(words_list):
            # Check what is the head of the word in the gold dependencies and its label
            for dep in gold_deps[w_ix]:
                if dep.target.idx == ix:
                    gold_head_ix = dep.source.idx
                    gold_label = dep.label
                    break
            # Check what is the head of the word in the predicted dependencies and its label
            for dep in pred_deps[w_ix]:
                if dep.target.idx == ix:
                    # Do the gold and predicted head match?
                    if dep.source.idx == gold_head_ix:
                        unlab_match += 1
                        # Does the label match? 
                        if dep.label == gold_label:
                            lab_match += 1
                    break
            total += 1

    return unlab_match/total, lab_match/total
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("[main] Dropout has to be a non-negative number strictly less than 1!")
    
    saveModelPath = "./models/"
    
    args.out = checkIfExists(args.out, isDir=True, createIfNotExists=True)
    if args.train:
        checkFile(args.train, fileExtension=".txt")
    if args.val:
        checkFile(args.val, fileExtension=".txt")
    if args.test:
        checkFile(args.test, fileExtension=".txt")
    if args.predict:
        checkFile(args.predict, fileExtension=".txt")
    _ = checkIfExists(saveModelPath, isDir=True, createIfNotExists=True)
    checkFile(args.pos, fileExtension=".txt")
    checkFile(args.tag, fileExtension=".txt")
    posToInd = readFile(args.pos)
    labelToInd = readFile(args.tag)
    posToInd = {w:i for i, w in enumerate(posToInd)}
    labelToInd = {w:i for i, w in enumerate(labelToInd)}
    _labelToInd = lambda label: labelToInd[label]
    indToLabel = {i:w for i, w in enumerate(labelToInd)}
    _indToLabel = lambda ind: indToLabel[ind]
    depToInd = {
        "NULL": 0,
    }
    for k in labelToInd.keys():
        depToInd[k.split("_")[-1]] = len(depToInd)

    if args.train:
        if not args.val:
            raise RuntimeError("[main] val must be provided along with train!")
        trainData = readFile(args.train)
        trainDataExtr = extractData(trainData)
        trainData, trainActions = processData(trainDataExtr, args.context, args.includeDep)
        trainDataLoader = createDataLoader(trainData, trainActions, labelToInd, posToInd, args.context, args.gloveName, args.gloveDim, args.batchSize, args.includeDep, depToInd)
        valData = readFile(args.val)
        valDataExtr = extractData(valData)
        valData, valActions = processData(valDataExtr, args.context, args.includeDep)
        valDataLoader = createDataLoader(valData, valActions, labelToInd, posToInd, args.context, args.gloveName, args.gloveDim, args.batchSize, args.includeDep, depToInd)
    if args.test:
        testData = readFile(args.test)
        testDataExtr = extractData(testData)
        testData, testActions = processData(testDataExtr, args.context, args.includeDep)
        testDataLoader = createDataLoader(testData, testActions, labelToInd, posToInd, args.context, args.gloveName, args.gloveDim, args.batchSize, args.includeDep, depToInd)
    if args.predict:
        predictData = readFile(args.predict)
        predictData = extractData(predictData)

    if torch.cuda.is_available:
        device = "cuda"
    else: 
        device = "cpu"
    logging.info("Using device:{}".format(device))

    if args.load:
        model = torch.load(f"{saveModelPath}model.pt")
        logging.info("Loaded model from {}model.pt".format(saveModelPath)) 
    else: 
        model = Actor(
            args.context, 
            args.gloveDim, 
            args.embedPos,
            args.hiddenDim, 
            posToInd, 
            labelToInd, 
            args.strategy, 
            device,
            args.includeDep,
            depToInd,
            args.embedLabel,
            args.dropout,
        )

    logging.info("Args: {}".format(args))

    executed = False
    datasetInst =  ActorDataset(
        labelToInd,
        posToInd,
        args.context,
        gloveName=args.gloveName,
        gloveDim=args.gloveDim,
        includeDep=args.includeDep,
        depToInd=depToInd,
    )
    if args.train:
        numTrainingSteps = args.numEpochs * len(trainDataLoader)
        maxSteps = args.maxSteps
        if maxSteps == -1:
            maxSteps = numTrainingSteps
        elif maxSteps > 0:
            maxSteps = math.ceil(maxSteps/len(trainDataLoader))
        else: 
            raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

        bestLearningRate = None
        bestValLAS = None
        bestValUAS = None
        bestValAcc = 0
        oriMaxSteps = maxSteps
        # saveModelAs = "{}model.pt".format(saveModelPath)
        saveModelAs = "{}model_{}_{}_{}_{}.pt".format(saveModelPath, args.gloveName, args.gloveDim, args.strategy, args.includeDep)
        for learningRate in args.learningRate:
            logging.info("Learning Rate: {}".format(learningRate))
            maxSteps = oriMaxSteps
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learningRate, 
                weight_decay=args.weightDecay,
                # amsgrad=True,
            )
            # optimizer = torch.optim.Adagrad(
            #     model.parameters(), 
            #     lr=learningRate, 
            #     weight_decay=args.weightDecay,
            #     # amsgrad=True,
            # )
            totalSteps = args.numEpochs
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                # num_warmup_steps=0.1*totalSteps,
                # num_warmup_steps=2000,
                num_warmup_steps=0,
                num_training_steps=totalSteps
            )
            scheduler = None
            weight=torch.ones(len(labelToInd))
            #Downweight "shift" action to tackle data imbalance
            weight[labelToInd["SHIFT"]] = 1/100
            lossFunction = torch.nn.CrossEntropyLoss(weight=weight).to(device)
            
            for epoch in range(args.numEpochs):
                curAcc, curLoss = trainModel(
                    model, 
                    trainDataLoader, 
                    lossFunction, 
                    optimizer, 
                    device, 
                    scheduler, 
                    maxSteps,
                    includeDep=args.includeDep
                )
                maxSteps -= len(trainDataLoader)
                valAcc, valLoss = evalModel(
                    model, 
                    lossFunction,
                    valDataLoader, 
                    device=device,
                    dataDesc="Validation batch", 
                    includeDep=args.includeDep
                )

                valWords, valLabels, valPreds = testModel(
                    model, 
                    valDataExtr, 
                    indToLabel, 
                    datasetInst, 
                    device, 
                    args.includeDep, 
                    dataDesc="Validation Data"
                ) 

                uas, las = compute_metrics(valWords, valLabels, valPreds, args.context)

                logging.info("Epoch {}/{}\nTraining Loss: {:0.2f}\nTrain Accuracy: {:0.2f}%\nValidation Loss: {:0.2f}\nValidation Accuracy: {:0.2f}%\nValidation UAS: {:0.2f}\nValidation LAS: {:0.2f}".format(epoch+1, args.numEpochs, curLoss, curAcc*100, valLoss, valAcc*100, uas, las))
                logging.info("*****")

                if not bestValLAS or bestValLAS <= las:
                    bestValLAS = las
                    bestValUAS = uas
                    bestValAcc = valAcc
                    bestLearningRate = learningRate
                    torch.save(model, saveModelAs)
                    logging.info("Model saved at '{}'".format(saveModelAs))
                if maxSteps <= 0:
                    break
        logging.info("Best learning rate: {}".format(bestLearningRate))
        logging.info("Best model's validation LAS: {:0.2f}".format(bestValLAS))
        logging.info("Best model's validation UAS: {:0.2f}".format(bestValUAS))
        logging.info("Best model's validation accuracy: {:0.2f}%".format(bestValAcc*100))

        model = torch.load(saveModelAs)
        executed = True
    if args.test or args.predict:
        if not args.train and not args.load:
            raise ValueError("[main] Either one of the following two inputs must be provided: (train) or (load)")
        if args.test:
            lossFunction = torch.nn.CrossEntropyLoss().to(device)
            testAcc, _ = evalModel(
                model, 
                lossFunction,
                testDataLoader, 
                device=device,
                dataDesc="Test batch", 
                includeDep=args.includeDep
            )

            testWords, testLabels, testPreds = testModel(
                model, 
                testDataExtr, 
                indToLabel, 
                datasetInst, 
                device, 
                args.includeDep, 
                dataDesc="Test Data"
            ) 

            uas, las = compute_metrics(testWords, testLabels, testPreds, args.context)

            logging.info("Test Accuracy: {:0.2f}%\nTest UAS: {:0.2f}\nTest LAS: {:0.2f}".format(testAcc*100, uas, las))
            logging.info("*****") 
        if args.predict:
            indToLabel = {v: k for (k,v) in labelToInd.items()}
            _, _, preds = testModel(
                model, 
                predictData, 
                indToLabel, 
                datasetInst, 
                device, 
                args.includeDep, 
                dataDesc="Predict Data"
            ) 
            writeFile([" ".join(p) for p in preds], args.out+args.predict.split("/")[-1])
        executed = True
    if not executed: 
        raise ValueError("[main] One of the following three inputs should be provided: (train, val, [[test], [predict], [load]]) or (load, test, [predict]) or (load, predict, [test])")
#----------------------------------------------------------------------------
if __name__=="__main__":
    main()