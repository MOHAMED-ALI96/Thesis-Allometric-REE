import os
from graphviz import Digraph

# Manually ensure Graphviz bin is accessible
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Output location
output_dir = r'E:\Thesis-Allometric-REE\plots\00_graphes'
dot = Digraph(format='png', directory=output_dir)

# Global Graph Settings
dot.attr(rankdir='BT', bgcolor='lightgray', style='filled', nodesep='0.5', ranksep='0.75')  # BT = bottom to top
dot.attr('node', shape='box', style='filled', fontname='Helvetica')

# Layer 5 - Bottom: Directory Iterators (brown)
dot.attr('node', fillcolor='saddlebrown', fontcolor='white')
dot.node('DIR_CT', 'DirectoryIteratorCTres\nDirectoryIterator')
dot.node('DIR_Atlas', 'DirectoryIteratorAtlasSEGres\nDirectoryIterator')
dot.node('DIR_SUV', 'DirectoryIteratorSUV\nDirectoryIterator')

# Layer 4 - itkImageFileReader
dot.attr('node', fillcolor='lightblue', fontcolor='black')
dot.node('CTres', 'CTres\nitkImageFileReader')
dot.node('AtlasSeg', 'AtlasSegOrgansMaskResampled\nitkImageFileReader')
dot.node('SUV', 'SUV\nitkImageFileReader')

# Layer 3 - OrganSeg
dot.node('OrganSeg', 'OrganSeg\nIntervalThreshold')

# Layer 2 - Arithmetic
dot.node('CTxOrgan', 'CTresXOrganSeg\nArithmetic')
dot.node('SUVxOrgan', 'SUVXOrganSeg\nArithmetic')

# Layer 1 - Top
dot.node('GetVol', 'GetVolume\nImageStatistics')
dot.node('GetSUV', 'GetSUVSum\nImageStatistics')

# Connections (Bottom → Top)
dot.edge('DIR_CT', 'CTres')
dot.edge('DIR_Atlas', 'AtlasSeg')
dot.edge('DIR_SUV', 'SUV')

dot.edge('CTres', 'CTxOrgan')
dot.edge('SUV', 'SUVxOrgan')
dot.edge('AtlasSeg', 'OrganSeg')

dot.edge('CTxOrgan', 'OrganSeg')
dot.edge('SUVxOrgan', 'OrganSeg')

dot.edge('CTxOrgan', 'GetVol')
dot.edge('SUVxOrgan', 'GetSUV')
dot.edge('OrganSeg', 'GetVol')
dot.edge('OrganSeg', 'GetSUV')

# Render to file
dot.render('organ_seg_pipeline', cleanup=True)
