def sel( d, indices ):
    return {k:v[indices] for k, v in d.items()}

def reduce_depth( node ):
    node.depth -= 1
    if hasattr( node, "right" ) and node.right is not None:
       reduce_depth( node.right ) 
    if hasattr( node, "left" ) and node.left is not None:
       reduce_depth( node.left ) 
