"""
Be sure you have the extra requirements installed.

>>> pip install requirements.extra.txt
"""

import requests
import networkx as nx
import minitorch
import visdom


## Create an autodiff expression here.
def expression():
    x = minitorch.Scalar(10, name="x")
    y = (x + 10.0) * 20
    y.name = "y"
    return y


class GraphBuilder:
    def __init__(self):
        self.op_id = 0
        self.hid = 0
        self.intermediates = {}

    def get_name(self, x):
        if not isinstance(x, minitorch.Variable):
            return "constant %s" % (x,)
        elif len(x.name) > 15:
            if x.name in self.intermediates:
                return "h%d" % (self.intermediates[x.name],)
            else:
                self.hid = self.hid + 1
                self.intermediates[x.name] = self.hid
                return "h%d" % (self.hid,)
        else:
            return x.name

    def run(self, final):
        queue = [[final]]

        G = nx.DiGraph()
        G.add_node(self.get_name(final))

        while queue:
            (cur,) = queue[0]
            queue = queue[1:]

            if cur.history is None:
                continue
            elif minitorch.is_leaf(cur):
                continue
            else:
                op = "%s (Op %d)" % (cur.history.last_fn.__name__, self.op_id)
                G.add_node(op)
                G.add_edge(op, self.get_name(cur))
                self.op_id += 1
                for input in cur.history.inputs:
                    G.add_edge(self.get_name(input), op)

                for input in cur.history.inputs:
                    if not isinstance(input, minitorch.Variable):
                        continue

                    seen = False
                    for s in queue:
                        if s[0] == input:
                            seen = True
                    if not seen:
                        queue.append([input])
        return G


def main():
    y = expression()
    G = GraphBuilder().run(y)
    nx.nx_pydot.write_dot(G, "tmp.dot")

    response = requests.get(
        "https://graphviz.gomix.me/graphviz",
        dict(layout="dot", format="svg", mode="download", graph=open("tmp.dot").read()),
    )
    out = open("tmp.svg", "w")
    out.write(response.text)
    out.close()
    vis = visdom.Visdom()
    vis.svg(response.text)


main()
