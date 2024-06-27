import ROOT
import ROOT

class Color(int):
    """Create a new ROOT.TColor object with an associated index"""

    def __new__(cls, r, g, b, name=""):
        self = int.__new__(cls, ROOT.TColor.GetFreeColorIndex())
        self.object = ROOT.TColor(self, r, g, b, name, 1.0)
        self.name = name
        return self

colors = [
    #Color(93, 125, 158, "cBlue"),
    #Color(202, 152, 56, "cYellow"),
    #Color(126, 163, 82, "cGreen"),
    #Color(174, 99, 82, "cRed"),
    #Color(177, 144, 110, "cBrown"),
    #Color(140, 129, 183, "cViolet"),

    Color(31, 119, 180, "cBlue"),      # Blue
    Color(255, 127, 14, "cOrange"),    # Orange
    Color(44, 160, 44, "cGreen"),      # Green
    Color(214, 39, 40, "cRed"),        # Red
    Color(148, 103, 189, "cPurple"),   # Purple
    Color(140, 86, 75, "cBrown"),      # Brown
    Color(227, 119, 194, "cPink"),     # Pink
    Color(127, 127, 127, "cGray"),     # Gray
    Color(188, 189, 34, "cOlive"),     # Olive
    Color(23, 190, 207, "cCyan"),      # Cyan
    Color(255, 187, 120, "cPeach"),    # Peach
    Color(197, 176, 213, "cLavender"), # Lavender
    Color(158, 218, 229, "cLightBlue"),# Light Blue
    Color(199, 199, 199, "cLightGray"),# Light Gray
    Color(152, 223, 138, "cLightGreen")# Light Green
]


#class Color(int):
#    """Create a new ROOT.TColor object with an associated index"""
#    __slots__ = ["object", "name"]
#
#    def __new__(cls, r, g, b, name=""):
#        self = int.__new__(cls, ROOT.TColor.GetFreeColorIndex())
#        self.object = ROOT.TColor(self, r, g, b, name, 1.0) 
#        self.name = name
#        return self
#
#
#colors = [Color(93, 125, 158, "cBlue"),
#          Color(202, 152, 56, "cYellow"),
#          Color(126, 163, 82, "cGreen"),
#          Color(174, 99, 82, "cRed"),
#          Color(177, 144, 110, "cBrown"),
#          Color(140, 129, 183, "cViolet"),
#        ]
#
for color in colors:
    setattr(ROOT, color.name, color)
