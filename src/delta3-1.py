#
# Copyright 2021, 2022 Violono UG (haftungsbeschränkt), Dr. Bernd Geiger
#
import datetime

#import en_core_web_lg

if datetime.date(2022, 3, 31) < datetime.date.today():
    print('texcompare.exe: demo period ended')
    exit(-1)

import numpy as np
import argparse
import traceback
import sys, types
from typing import *
import io
import logging
import datetime
import regex as re
import spacy

from datetime import date
import sys
import io
from io import StringIO
# import torch
#from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
#from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
#from spacy.util import compile_infix_regex
#import os, sys, codecs
#import logging
#from datasets import set_caching_enabled

#set_caching_enabled(True)
# from spacy_language_detection import LanguageDetector
#nlpEN =  en_core_web_lg.load()
nlpEN = spacy.load('en_core_web_lg')

parser = argparse.ArgumentParser()
parser.add_argument('f0', type=str)
parser.add_argument('f1', type=str)
args = parser.parse_args()


if len( vars(args) ) != 2:
    print('usage: textcompare file1 file2')

try:
    fi0 = open(args.f0, 'r', encoding='utf-8')
except:
    print('textcompare error: ', fi0, ' not found')
    exit(-1)

try:
    fi1 = open(args.f1, 'r', encoding='utf-8')
except:
    print('textcompare error: ', fi1, ' not found')
    exit(-1)

try:
    t0 = fi0.read()
    fi0.close()
except:
    print('textcompare error: ', fi0, ' cannot be read')
    exit(-1)

try:
    t1 = fi1.read()
    fi1.close()
except:
    print('textcompare error: ', fi1, ' cannot be read')
    exit(-1)


print("""
*************************************************************************
***                          text compare demo                        ***
***                                                                   ***
***                Copyright 2022,  semafora systems GmbH             ***
*************************************************************************
""")


#ferr = open('textcompare.err', 'a', encoding='utf-8')


class fstyle:
    HEADER = '\033[95m'
    BLUE = '\033[96m'
    #        BLUE = '\033[1m\033[38;5;33m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    #        RED = '\033[9m\u001b[31m'
    RED = '\033[9m\033[31m'
    FAIL = '\033[91m'
    CLEAR = '\u001b[0m'
    #        CLEAR = ' \u001b[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def GetFStyle(style):
    return f"{style}"


def deltaWords(l0, l1):


    def normS(s):
        s = re.sub(r'(?<=[A-Za-z0-9])\s([\!\#\$\%\&\'\*\+\-\.,;\^\_\`\|\~\:])', r"\1", s)
        s = re.sub(r'(?<=[A-Za-z0-9])\s([\!\#\$\%\&\'\*\+\-\.,;\^\_\`\|\~\:\[\]\(\)\{\}])', r"\1", s)
        return s

    def GetDelta(l0, l1, tag):
        seq = {}
        seqT1 = {}
        seqT = {}
        iD = 0

        # remove smaller double backward associations

        for i in range(len(l1)):
            for j in range(i, len(l1)):
                tmp = l1[i:j + 1]
                for i1 in range(len(l0)):
                    for j1 in range(i1, len(l0)):
                        if tmp == l0[i1:j1 + 1]:
                            seqT[i] = (tmp, i, j, i1, j1, 'i')

        seqT1 = seqT.copy()
        #        print(seq)

        # remove smaller double forward associations
        for k0, v0 in seqT.items():
            for k1, v1 in seqT.items():
                if len(v0[0]) < len(v1[0]) and v1[3] <= v0[3] <= v1[4]:
                    if k0 in seqT1:
                        seqT1.pop(k0)
                if len(v0[0]) < len(v1[0]) and v1[1] <= v0[1] <= v1[2]:
                    if k0 in seqT1:
                        seqT1.pop(k0)

        seq = seqT1.copy()
        l1AS = set(range(len(l1)))  # get all possible indices
        l1BS = set()

        for k, v in seqT1.items():  # get all target indices used
            for i in range(v[1], v[2] + 1):
                l1BS.add(i)

        l1S = l1AS.difference(l1BS)  # get all indices not associated with other list

        #    print(l1S)

        for i in l1S:
            try:
                seq[i] = ([l1[i]], i, i, False, False, tag)
            except:
                print("***error in dict gen after set difference")
                exit(-1)
        #    print(l1S)
        seqD = {key: seq[key] for key in sorted(seq.keys())}  # sort the dict
        ind = [k for k in seqD]

        minI = min(ind)
        maxI = max(ind)

        seqD1 = {}
        for i in range(len(ind)):  # add prev / next dict element info
            tT = seqD[ind[i]]
            if ind[i] == minI:
                seqD1[ind[i]] = (*tT, [minI, ind[i + 1]])
            elif ind[i] == maxI:
                seqD1[ind[i]] = (*tT, [ind[i - 1], maxI])
            else:
                seqD1[ind[i]] = (*tT, [ind[i - 1], ind[i + 1]])

        return seqD1, set(ind)

    #    return dict(sorted(seq.items(),key=))

    seq0, s0S1 = GetDelta(l1, l0, 'o')
    #    print(seq0)
    seq1, s1S1 = GetDelta(l0, l1, 'n')
    #    print(seq1)

    # exit(0)
    end = False
    started = False
    l3 = []
    l3T = []
    s0S = set()
    s0Si = set()

    for k1, v1 in seq1.items():
        if v1[5] == 'n':  # insert first the new ones in target
            for e in v1[0]:  # always split "i" word lists and put the words in separate list elements
                l3.append([e, v1[5]])
            if k1 == v1[6][1]:  # if reach the end of seq1 with a new element at the end check for leftovers in seq0
                s0SiL = list(s0S1.difference(s0S))[::-1]  # really reverse???
                for s1 in s0SiL:
                    for e in seq0[s1][0]:
                        l3.append([e, seq0[s1][5]])
            continue  # n ist done at this step - to for next step
        elif v1[5] == 'i':  # before filling up with identical first check if there are old ones to fill up
            k0 = seq0[v1[3]][6][0]
            l3T = []
            while True:
                try:
                    sq = seq0[k0]
                    if sq[5] != 'i':
                        started = True
                        s0S.add(k0)  # note what will be processed here
                        for e in sq[0][
                                 ::-1]:  # we are searwching backwards - so store inverted (should be only one element in though
                            l3T.append([e, sq[5]])
                        if k0 == sq[6][0]:
                            for e1 in l3T[::-1]:
                                l3.append(e1)  # end of backwards move
                            started = False
                            break  # no further prev element
                        else:
                            k0 = sq[6][0]
                    else:  # seq0 is also an 'i'
                        if started:
                            started = False
                            for e1 in l3T[::-1]:
                                l3.append(e1)  # end of backwards move
                        break
                except:
                    print('error in backward ind set collection')
                    exit(0)
            for e in v1[0]:  # add all the i words
                l3.append([e, v1[5]])
            s0S.add(v1[3])  # note the index of the corresponding i words in the ori sentence (just one dict elem)

            if k1 == v1[6][1]:  # reached the end of sq1
                s0SiL = list(s0S1.difference(s0S))[::-1]
                for s1 in s0SiL:
                    for e in seq0[s1][0]:
                        for e1 in l3T:
                            l3.append(e1)  # end of backwards move
                        # l3.append([e, seq0[s1][5]])
            continue
    t = ''
    for e in l3:
        try:
            if e[1] == 'n':
                t += ' ' + GetFStyle(fstyle.BLUE) +e[0]+GetFStyle(fstyle.CLEAR)
            elif e[1] == 'o':
                t += ' ' + GetFStyle(fstyle.RED) + e[0]+GetFStyle(fstyle.CLEAR)
            elif e[1] == 'i':
                #                t += GetFStyle(fstyle.CLEAR) + ' ' + e[0]+GetFStyle(fstyle.CLEAR)
                t += ' ' + e[0]
        except:
            print('coloring went wrong')
    t = normS(t)

    return t


# spacy.load('de_dep_news_trf')
# prepare to detect language
# Language.factory("language_detector", func=get_lang_detector)
# nlpEN.add_pipe('language_detector', last=True)

def DeltaInd(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))


def ovL(l0, l1):
    return [(l0, i, i + len(l0) - 1) for i in range(len(l1) - len(l0) + 1) if l1[i:i + len(l0)] == l0]


def eqL(l0, l1):
    l0.append('')
    l1.append('')
    len0 = len(l0)
    len1 = len(l1)
    lt = 0
    if len0 > len1:
        lt = len0
        l1.extend(['' for x in range(len0 - len1)])
    elif len0 < len1:
        lt = len1
        l0.extend(['' for x in range(len1 - len0)])
    else:
        lt = len1
    return l0, l1, lt


def DeltaEq(l0, l1):
    i = 0
    ind = [-1]
    comp = {}
    ended = False
    lt = len(l0)
    while (i < lt and l0[i] != ''):
        save = ()
        jc = 0
        for j in range(i + 1, lt):
            #        if j <= i: continue
            tmp = ovL(l0[i:j], l1)
            #        print(i, j, tmp)
            if tmp == []:
                #            pass
                if j == (i + 1):
                    i += 1
                else:
                    i += jc
                    for jj, ii in enumerate(range(save[0][1], save[0][2] + 1)):
                        #                    print(jj,ii,save[0][0][jj])
                        #                    print(e,ii + save[0][2])
                        comp[(ii, 0)] = [save[0][0][jj]]
                        ind.append(ii)
                #                print(comp)
                # print(save)
                ended = True
                break
            else:
                save = tmp
                jc += 1
        if not ended:
            ended = False
            i += 1
    ind.append(lt)
    return comp, ind


def sntComp(l0, l1):
    l0, l1, lt = eqL(l0, l1)
    comp0, ind0 = DeltaEq(l0, l1)
    comp1, ind1 = DeltaEq(l1, l0)
    invInd0 = DeltaInd(ind0)
    invInd1 = DeltaInd(ind1)

    # print(comp0,invInd0)
    # print(comp1,invInd1)

    comp0N = {}
    comp1D = {}

    for e in invInd0:
        if l1[e] != '':
            comp0N[(e, 1)] = [l1[e]]

    for e in invInd1:
        if l0[e] != '':
            comp1D[(e, -1)] = [l0[e]]

    comp0.update(comp0N)
    comp0.update(comp1D)

    comp = {key: comp0[key] for key in sorted(comp0)}

    sent = ''
    for k, v in comp.items():
        # print(k,v)
        if k[1] == -1:
            sent += GetFStyle(fstyle.RED)
            sent += v[0] + ' '
            sent += GetFStyle(fstyle.CLEAR)
        elif k[1] == 1:
            sent += GetFStyle(fstyle.BLUE)
            sent += v[0] + ' '
            sent += GetFStyle(fstyle.CLEAR)
        elif k[1] == 0:
            sent += v[0] + ' '
    return sent


def normS(s):
    s = re.sub(r'(?<=[A-Za-z0-9])\s([\!\#\$\%\&\'\*\+\-\.,;\^\_\`\|\~\:])', r"\1", s)
    s = re.sub(r'(?<=[A-Za-z0-9])\s([\!\#\$\%\&\'\*\+\-\.,;\^\_\`\|\~\:\[\]\(\)\{\}])', r"\1", s)


def ListPos(l, s, e):
    l1 = list(map(lambda st: len(st), l))


def lstOverlap(l0, l1):
    def f(seq, n):
        return [seq[max(i, 0):i + n] for i in range(-n + 1, len(seq))]

    t0 = [tuple(x) for x in f(l0, 2)[1:][:-1]]
    t1 = [tuple(x) for x in f(l1, 2)[1:][:-1]]
    if t0 == []: t0.append('')
    if t1 == []: t1.append('')

    setA = set(t0)
    setB = set(t1)

    overlap = setA & setB
    universe = setA | setB
    try:
        return (float(len(overlap)) / len(setB), float(len(overlap)) / len(setA))
    except:
        pass
        return (0, 0)


def eqlSL(sntL0, sntL1):
    ### adjust length of sentence list
    len0 = len(sntL0)
    len1 = len(sntL1)
    lt = 0
    if len0 > len1:
        lt = len0
        sntL1.extend([' ' for x in range(len0 - len1)])
    elif len0 < len1:
        lt = len1
        sntL0.extend([' ' for x in range(len1 - len0)])
    else:
        lt = len1

    return sntL0, sntL1, lt





splitSent = True

t0 = """
Four identical tyres shall be fitted on the test vehicle. In the case of tyres with a load capacity index in excess of 121 and without any dual fitting indication, two of these tyres of the same type and range shall be fitted to the rear axle of the test vehicle; the front axle shall be fitted with tyres of size suitable for the axle load and planed down to the minimum depth in order to minimize the influence of tyre/road contact noise while maintaining a sufficient level of safety. Winter tyres that in certain Contracting Parties may be equipped with studs intended to enhance friction shall be tested without this equipment. Tyres with special fitting requirements shall be tested in accordance with these requirements (e.g. rotation direction). The tyres shall have full tread depth before being run-in.
Tyres are to be tested on rims permitted by the tyre manufacturer.
"""


t1 = """
Four identical tyres shall be fitted on the test vehicle. In the case of C3 tyres with a load capacity index in excess of 121 and without any dual fitting indication, two of these tyres of the same type and range shall be fitted to the rear axle of the test vehicle; the front axle shall be fitted with tyres of size suitable for the axle load and planed down to the minimum depth in order to minimize the influence of tyre/road contact noise while maintaining a sufficient level of safety.
In the case of C2 tyres with a load capacity index lower or equal to 121, with a section width wider than 200 mm, an aspect ratio lower than 55, a rim diameter code lower than 15 and without any dual fitting indication, two of these tyres of the same type and range shall be fitted to the rear axle of the test vehicle; the front axle shall be fitted with tyres of a size suitable for the axle load and planed down to the minimum depth in order to minimize the influence of tyre/road contact noise while maintaining a sufficient level of safety.
Tyres with special fitting requirements shall be tested in accordance with these requirements (e.g. rotation direction). The tyres shall have full tread depth before being run-in.
Tyres are to be tested on rims permitted by the tyre manufacturer.
"""

t0 = t0.replace('\n', ' ').strip()
t1 = t1.replace('\n', ' ').strip()

# text = list(map(lambda st: str.replace(st, '\n', ' ').strip(), text))

doc0 = nlpEN(t0)
doc1 = nlpEN(t1)

# attach a newline at the end

# sntL0 = list(map(lambda st: str(st)+ '\n', doc0.sents))
# sntL1 = list(map(lambda st: str(st)+ '\n', doc1.sents))

# get the single sentences

sntL0 = list(map(lambda st: str(st), doc0.sents))
sntL1 = list(map(lambda st: str(st), doc1.sents))

sntL0, sntL1, lt = eqlSL(sntL0, sntL1)

# Generated lists of tokenized words
w0 = []
w1 = []

for e in sntL0:
    w0.append([token.orth_ for token in nlpEN.tokenizer(e)])
for e in sntL1:
    w1.append([token.orth_ for token in nlpEN.tokenizer(e)])

# check what


sCor0 = np.zeros((lt, lt))
sCor1 = np.zeros((lt, lt))

for i, e in enumerate(w0):
    for j in range(len(w1)):
        res = lstOverlap(e, w1[j])
        sCor0[j, i] = res[0]
        sCor1[j, i] = res[1]
        #        sCor[j, i] = res[1]
#        print(i, j, res)

indS = set()
indS.add(-1)
sentL = []
maxV = np.amax(sCor0, axis=0)
indV = np.argmax(sCor0, axis=0)

for i in range(lt):
    o = maxV[i]
    ind = indV[i]
    if not "".join(w0[i]).strip(): continue  # empty strings are not processed
    if o <= 0.5:  # case indipendent single
        sentL.append([[w0[i]], -1, [i]])  # -1 stands for deleted
    elif o == 1.0:  # case identical
        indS.add(ind)  # add index of processed new sent into list
        for j in range(ind):
            if j not in indS:
                if not "".join(w1[j]).strip(): continue
                sentL.append([[w1[j]], 1, [j]])
                indS.add(j)  # mark as processed
        sentL.append([[w0[i]], 0, [i]])  # 0 stands for unchanged
    elif 0.5 < o < 1.0:  # case changes in sentence to be processed
        indS.add(ind)  # add index of deployed new sent into list
        for j in range(ind):
            if j not in indS:
                if not "".join(w1[j]).strip(): continue
                sentL.append([[w1[j]], 1, [j]])  # 1 stands for new
                indS.add(j)  # mark as processed
        sentL.append([[w0[i], w1[ind]], 0.5, [i, ind]])  # 0.5 stands for modified
    else:
        print('not possible: ', o)
        exit(-1)

for j in range(lt):  # get leftovers in new sent
    if not "".join(w1[j]).strip(): continue  # empty strings are not processed
    if j not in indS:
        sentL.append([[w1[j]], 1, [j]])  # 1 stands for new
        indS.add(j)  # mark as processed

sentLT = []
try:
    for i, e in enumerate(sentL):
        t = ''
        if e[1] == -1:
            t = GetFStyle(fstyle.RED) + sntL0[e[2][0]] + GetFStyle(fstyle.CLEAR)
            sentLT.append(t.strip() + '\n')
        elif e[1] == 0:
            t = GetFStyle(fstyle.CLEAR) + sntL0[e[2][0]] + GetFStyle(fstyle.CLEAR)
            sentLT.append(t.strip() + '\n')
        elif e[1] == 0.5:
            #            t = sntComp(w0[e[2][0]],w1[e[2][1]])
            t = deltaWords(w0[i], w1[i])
            # t = GetFStyle(fstyle.YELLOW) + sntL0[e[2][0]] + GetFStyle(fstyle.CLEAR) + GetFStyle(fstyle.GREEN) + sntL1[e[2][1]] + GetFStyle(fstyle.CLEAR)
            sentLT.append(t.strip() + '\n')
        elif e[1] == 1:
            t = GetFStyle(fstyle.BLUE) + sntL1[e[2][0]] + GetFStyle(fstyle.CLEAR)
            sentLT.append(t.strip() + '\n')
except:
    print('coloring went wrong')
    exit(-1)

fout = open('text_compare.txt', 'w', encoding='ansi')



for i, e in enumerate(sentLT):
    print(i, e, file = fout)
    print(i, e)

fout.close()
