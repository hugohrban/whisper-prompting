#!/usr/bin/env python3
import jiwer
import sys
import edlib
import re


def load_file(fn):
    with open(fn, "r") as f:
        return f.read()


def load_ortots(fn):
    ts = []
    words = []
    with open(fn, "r") as f:
        for line in f:
            a, b, word = line.split(" ")
            #            print(a,b,word)
            ts.append((int(a), int(b)))
            words.append(word.strip())
    return ts, words


# A = load_file(sys.argv[1])

transformation = jiwer.Compose(
    [
        jiwer.SubstituteRegexes({r"[-'/]": " "}),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        #    jiwer.SentencesToListOfWords(word_delimiter=" "),
    ]
)


def open_transform(gold_fn, asr_fn):
    ts, orto = load_ortots(gold_fn)

    B = load_file(asr_fn)

    A = " ".join(orto)

    tA = transformation(A)
    tB = transformation(B)
    return tA, tB, ts


def eprint(*a, **kw):
    print(*a, **kw, file=sys.stderr)


def find_iter(p, x):
    return [m.start() for m in re.finditer(p, x)]


def swap_chars(i, j, x):
    ci, cj = x[i], x[j]
    l = list(x)
    l[i] = cj
    l[j] = ci
    return "".join(l)


def nice_alignments(A, B):
    A, B = A.strip(), B.strip()
    a = edlib.align(A, B, task="path")
    na = edlib.getNiceAlignment(a, A, B)

    # I want to fix this:
    # the egyptian / --e--gyptian ==> ----egyptian

    #    for k,v in na.items():
    #        print(k,v)

    aA, aB = na["query_aligned"], na["target_aligned"]
    m = na["matched_aligned"]

    for i in find_iter(r"-\|-", m):
        j = i + 1  # the | char index

        c = aA[j]  # the char that should be moved

        k = j + 1
        while k < len(m):
            #            print(k, m[k])
            if m[k] == "|":
                break
            k += 1
        # k-1: the position of c char (possibly)
        if aA[k - 1] == c or aB[k - 1] == c:
            #            print("move")
            aA = swap_chars(j, k - 1, aA)
            aB = swap_chars(j, k - 1, aB)
            m = swap_chars(j, k - 1, m)

    #        print(i, m[i-3:i+3])

    na["query_aligned"] = aA
    na["target_aligned"] = aB
    na["matched_aligned"] = m

    return na


def rep(x):
    return x.replace("-", "").strip()


def process_asr_align(tA, tB, na, debug_prints=True):
    ta_al = na["query_aligned"]
    if debug_prints:
        eprint(tA)
        eprint("---")
        eprint(tB)

        # print(list(na.keys()))
        eprint()

        i = 0
        while i < len(ta_al):
            eprint(na["matched_aligned"][i : i + 120])
            eprint(ta_al[i : i + 120])
            eprint(na["target_aligned"][i : i + 120])
            eprint()
            i += 120

    a = ta_al
    b = na["target_aligned"]

    out = []

    def it(i, j):
        aw = a[j:i]
        bw = b[j:i]
        if "-" not in aw and "-" not in bw:
            if aw == bw:
                out.append((aw, "C", bw))
            #    print(aw,"->",bw, "C")
            else:
                out.append((aw, "S", bw))
            #    print(aw,"->",bw, "S")
        else:
            raw = rep(aw)
            rbw = rep(bw)
            if raw == rbw:
                c = "C"
            elif raw in rbw:
                c = "I"
            elif not rbw:
                c = "D"
            else:
                c = "S"
            out.append((raw, c, rbw))
        j = i + 1
        return j

    j = 0
    for i in range(len(ta_al)):
        if a[i] == " ":
            j = it(i, j)
    it(len(ta_al), j)
    return out


# if __name__ == "__main__":

# #    tA, tB, ts = open_transform(sys.argv[1], sys.argv[2])
#     with open(sys.argv[1], "r") as f:
#         tA = " ".join(f.read().lower().split())
#     with open(sys.argv[2], "r") as f:
#         tB = " ".join(f.read().lower().split())

#     tA = transformation(tA)
#     tB = transformation(tB)

#     na = nice_alignments(tA, tB)
#     out = process_asr_align(tA, tB, na)
# #    for x in out:
# #        print(*x)
