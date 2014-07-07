import igraph as ig
import operator
from collections import defaultdict
from time import sleep


#############################
# Fuzzy Modularity Measures #
#############################

def nepusz_modularity(G, cover):
    raise NotImplementedError("See the CONGA 2010 paper")

def zhang_modularity(G, cover):
    raise NotImplementedError("""See 'Identification of overlapping community structure in
        complex networks using fuzzy C-means clustering'""")

def nicosia_modularity(G, cover):
    raise NotImplementedError("""See 'Extending the definition of
        modularity to directed graphs with overlapping communities'""")


#############################
# Crisp modularity measures #
#############################

def count_communities(G, cover):
    """
    Helper for lazar_modularity.

    Returns a dict {v:count} where v is a vertex id and
    count is the number of different communities it is
    assigned to.
    """
    counts = {i.index : 0 for i in G.vs}
    for community in cover:
        for v in community:
            counts[v] += 1
    return counts


def count_internal_edges(G, vertex, cluster):
    """
    Given a graph, a vertex in the graph, and a list of vertex
    ids "cluster", returns a set of edge ids that are internal
    to the cluster, as well as the difference between the number of
    internal edges and external edges.
    """
    internal = set()
    internalExternalCount = 0
    for edge in G.es[G.incident(vertex)]:
        if edge.tuple[1] in cluster:
            internal.add(edge)
            internalExternalCount += 1
        else: internalExternalCount -= 1
    return internal, internalExternalCount


def lazar_modularity(G, cover):
    """
    Returns the crisp modularity measure as defined by Lazar et al. 2009

    Defined as the average edge density times normalized difference
    between inter and intracommunity edges for each community.

    See CONGA 2010 or Lazar's paper for a precise definition.
    """
    counts = count_communities(G, cover)
    totalModularity = 0
    for cluster in cover:
        clusterSet = set(cluster)
        edgeSet = set()
        totalNormalizedDiff = 0
        for vertex in cluster:
            neighbors = G.neighbors(vertex)
            internalEdges, internalExternalCount = count_internal_edges(G, vertex, clusterSet)

            # Counting internal edges in a set so we don't repeat.
            edgeSet |= internalEdges

            # Normalizing by the vertex's degree and the number of communities it is in.
            normalization = G.vs[vertex].degree() * counts[vertex]
            totalNormalizedDiff += (internalExternalCount)/float(normalization)

        # Multiplying by the edge density.
        numEdges = float(len(edgeSet))
        numVertices = float(len(cluster))
        try:
            edgeDensity = numEdges / ((numVertices * numVertices-1) / 2)
        except ZeroDivisionError:
            # Not well defined for single vertex communities.
            # We can assume it is a bad cover, however,
            # so we return a 0 modularity.
            return 0
        edgeDensity *= 1 / numVertices
        totalModularity += totalNormalizedDiff * edgeDensity

    return totalModularity / len(cover)




##################################
# Classes for overlapping covers #
##################################

class CrispOverlap(object):
    """
    TODO
    """
    def __init__(self, graph, covers, modularities=None, optimal_count=None, modularity_measure="lazar"):
        """
        Initializes a CrispOverlap object with the given parameters.

            Graph: The graph to which the object refers
            covers: a dict of VertexCovers, also referring to this graph, of the form {k : v}
                where k is the number of clusters and v is the vertexCluste
            modularities (optional): a dict of modularities of the form {c:m} where c is
                the number of clusters and m is the modularity.
            optimal_count (optional): A hint for the number of clusters to use.
            modularity_measure (optional): The name of the modularity function to use.
                Right now, the only choice is "lazar."
        """
        # Possibly figure out a better data structure like a merge
        # list that contains all information needed?

        # So far only know of Lazar's measure for crisp overlapping.
        self._measureDict = {"lazar" : lazar_modularity}
        self._covers = covers
        self._graph = graph
        self._optimal_count = optimal_count
        self._modularities = modularities
        if modularity_measure in self._measureDict:
            self._modularity_measure = modularity_measure
        else: raise KeyError("Modularity measure not found.")


    def __getitem__(self, numClusters):
        """
        Returns the cover with the given number of clusters.
        """
        if not numClusters:
            raise KeyError("Number of clusters must be a positive integer.")
        return self._covers[numClusters]

    def __iter__(self):
        """
        Iterates over the covers in the list.
        """
        return (v for v in self._covers.values())

    def __len__(self):
        """
        Returns the number of covers in the list.
        """
        return len(self._covers)

    def __nonzero__(self):
        """
        Returns True when there is at least one cover in the list.
        """
        return bool(self._covers)


    def __str__(self):
        """
        Returns a string representation of the list of covers.
        """
        return '{0} vertices in {1} possible covers.'.format(len(self._graph.vs), len(self._covers))


    def as_cover(self):
        """
        Returns the optimal cover (by modularity) from the object.
        """
        return self._covers[self.optimal_count]


    def change_modularity_measure(measure, recalculate=True):
        """
        Given measure, the name of a new modularity measure, switches
        the modularity function used. If recalculate=True, also recalculates
        the modularities and optimal count.

        Note: currently useless, as there is only one available measure.
        """
        if measure in self._measureDict:
            self._modularity_measure = measure
        else: raise KeyError("Modularity measure not found.")
        if recalculate:
            self.recalculate_modularities()

    def recalculate_modularities(self):
        """
        Recalculates the modularities and optimal count using the modularity_measure.
        """
        modDict = {}
        for cover in self._covers.itervalues():
            modDict[len(cover)] = self._measureDict[self._modularity_measure](self._graph, cover)
        self._modularities = modDict
        self._optimal_count = max(self._modularities.iteritems(), key=operator.itemgetter(1))[0]
        return self._modularities


    @property
    def modularities(self):
        """
        Returns the a dict {c : m} where c is the number of clusters
        in the cover and m is the modularity. If modularity has not
        been calculated, it recalculates it for all covers. Otherwise,
        it returns the stored dict.

        Note: Call recalculate_modularities to recalculate the modularity.
        """
        if self._modularities:
            return self._modularities
        self._modularities = self.recalculate_modularities()
        return self._modularities


    @property
    def optimal_count(self):
        """Returns the optimal number of clusters for this dendrogram.

        If an optimal count hint was given at construction time and
        recalculate_modularities has not been called, this property simply returns the
        hint. If such a count was not given, this method calculates the optimal cover
        by maximizing the modularity along all possible covers in the object.

        Note: Call recalculate_modularities to recalculate the optimal count.
        """
        if self._optimal_count is not None:
            return self._optimal_count
        else:
            modularities = self.modularities
            self._optimal_count = max(modularities.items(), key=operator.itemgetter(1))[0]
            return self._optimal_count


    def make_fuzzy(self):
        """
        TODO. see CONGA 2010
        """
        pass




# TODO. Other algorithms like FOG return a fuzzy overlapping.

# Nothing below this line has been implemented.
###############################################

class FuzzyOverlap(object):
    """
    TODO
    """
    def __init__(self, graph, coverDict, optimal_count=None, modularity_measure = "nepusz"):
        """
        TODO
        """
        # this repeats a ton of data. figure out a better data structure.
        self._coverDict = coverDict
        self._graph = graph
        self._optimal_count = optimal_count
        if modularity_measure in modularityDict:
            self._modularity_measure = modularity_measure
        else: raise KeyError("Modularity measure not found.")
        self._mod_flag = False
        self.fuzzyDict = {"nepusz" : nepusz_modularity,
                          "zhang": zhang_modularity,
                          "nicosia" : nicosia_modularity}


    @property
    def optimal_count(self):
        """
        TODO
        """
        if self._optimal_count is not None and not mod_flag:
            return self._optimal_count
        else:
            max_mod, max_index = max(enumerate(self.list_modularities))
            mod_flag = False
            return max_index + 1



    def change_modularity_measure(self, new):
        mod_flag = True
        if new in modularityDict:
            self._modularity_measure = new
        else: raise KeyError("Modularity measure not found.")