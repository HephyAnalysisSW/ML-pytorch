''' Base class to carry class attributes'''

class NodeBase:
    def set_class_attrs( self, **kwargs ):
        for key, val in kwargs.items():
            if hasattr( self.__class__, key ):
                print ("Warning! Overwriting global NodeBase attribute: %s"%key)
            setattr( NodeBase, key, val )
            #print ("Set global attribute", NodeBase, key, val)

    cache = []
    @staticmethod
    def add_instance( instance ):
        NodeBase.cache.append( hex(id(instance)) )
        
    @staticmethod
    def remove_instance( instance ):
        NodeBase.cache.remove( hex(id(instance)) )
