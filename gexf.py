from datetime import date
from lxml import etree

class Node:
    def __init__(self, id : str, label : str=None) -> None:
        self.id = id
        self.label = label
        
class Edge:
    def __init__(self, source : str | Node, target : str | Node, id : str=None, edge_type : str=None, weight : float=1.0, timestamp : int=None) -> None:
        self.id = id
        self.edge_type = edge_type 
        self.weight = weight
        self.timestamp = timestamp
        self.source = source
        if isinstance(source, Node):
            self.source = source.id
        self.target = target
        if isinstance(target, Node):
            self.target = target.id

class Graph:
    def __init__(self, edge_type : str='undirected', mode : str='static') -> None:
        self.edge_type = edge_type
        self.mode = mode
        if self.mode == 'dynamic':
            self.time_representation = 'timestamp'
            self.time_format = 'integer'
        self.nodes = []
        self.edges = []

    def add_node(self, id : str=None, label : str=None, node : Node=None) -> None:
        if node is None:
            self.nodes.append(Node(id, label))
        else:
            self.nodes.append(node)

    def add_edge(self, source : str | Node = None, target : str | Node = None, id : str=None, edge_type : str=None, weight : float=None, timestamp : int=None, edge : Edge=None) -> None:
        if edge is None:
            self.edges.append(Edge(source, target, id, edge_type, weight, timestamp))
        else:
            self.edges.append(edge)

class Gexf:
    def __init__(self, creator : str=None, description : str=None) -> None:
        self.creator = creator
        self.description = description
        self.xmlns = 'http://gexf.net/1.3'
        self.xsi = 'http://www.w3.org/2001/XMLSchema-instance'
        self.schema_location = 'http://gexf.net/1.3 http://gexf.net/1.3/gexf.xsd'
        self.version = 1.3
        self.last_modified_date = str(date.today())

    def load_graph(self, graph : Graph) -> None:
        self.graph = graph
        namespace = '{%s}' % self.xmlns
        nsmap = {None : self.xmlns}
        self.root = etree.Element(namespace + 'gexf', nsmap=nsmap)
        meta = etree.SubElement(self.root, 'meta', lastmodifieddate=self.last_modified_date)
        if self.creator is not None:
            creator = etree.SubElement(meta, 'creator')
            creator.text = self.creator
        if self.description is not None:
            description = etree.SubElement(meta, 'description')
            description.text = self.description
        graph_element = etree.SubElement(self.root, 'graph', defaultedgetype=graph.edge_type, mode=graph.mode)
        if graph.mode == 'dynamic':
            graph_element.attrib['timerepresentation'] = graph.time_representation
            graph_element.attrib['timeformat'] = graph.time_format
        nodes = etree.SubElement(graph_element, 'nodes')
        edges = etree.SubElement(graph_element, 'edges')
        for node in graph.nodes:
            node_element = etree.SubElement(nodes, 'node', id=str(node.id))
            if node.label is not None:
                node_element.attrib['label'] = node.label
        for edge in graph.edges:
            edge_element = etree.SubElement(edges, 'edge', source=str(edge.source), target=str(edge.target))
            if edge.id is not None:
                edge_element.attrib['id'] = str(edge.id)
            if edge.edge_type is not None:
                edge_element.attrib['type'] = edge.edge_type
            if edge.weight is not None:
                edge_element.attrib['weight'] = str(edge.weight)
            if edge.timestamp is not None:
                edge_element.attrib['timestamp'] = str(edge.timestamp)

    def __str__(self) -> str:
        if self.root is None:
            return ''
        return etree.tostring(self.root, pretty_print=True).decode()
