class DecisionNode:

    def __init__( self, indices ):
        self.indices = indices
        self.left, self.right = None, None 

    def print_tree(self, _depth=0):
        print('%sDecisionNode' % (_depth* ' ') ) 
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)
