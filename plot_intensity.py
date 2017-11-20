OpenDatabase("intensities.txt")
fop = {"Lines to skip at beginning of file": 0,
       "Column for X coordinate (or -1 for none)": 0,
       "Column for Y coordinate (or -1 for none)": 1,
       "Column for Z coordinate (or -1 for none)": 2,
       "Data layout": 0,
       "First row has variable names": 0}
SetDefaultFileOpenOptions("PlainText", fop)
ReOpenDatabase("intensities.txt")

AddPlot("Pseudocolor", "var03")

p = PseudocolorAttributes()
p.scaling = p.Log
SetPlotOptions(p)

DrawPlots()
