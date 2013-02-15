# coding=utf-8

from math import log,sqrt
from random import randint

# debug utils    
debug_ids = set()

def debugon(*id_list):
    global debug_ids
    debug_ids = debug_ids.union(set(id_list))
    return debug_ids

def undebug(*id_list):
    global debug_ids
    if id_list:
        debug_ids = debug_ids.difference(set(id_list) )
    else:
        debug_ids = set()

def dbg(dbg_id,pattern=None,*values):
    if pattern and dbg_id in debug_ids:
        print pattern % values
        
    return dbg_id in debug_ids
    
# A words cluster algorithm
class CharWeight(dict):
    """Use IDF(Inverse Document Frequency) as char's weight"""
    
    def __init__(self,words):
        for w in words:
            for c in set(w):
                self[c] = self.get(c,0) + 1
                
        n = len(self)

        for c in self.iterkeys():
            self[c] = log(float(n) / self[c],2)

    def __call__(self, char):
        return self.get(char,0)

def read_words(word_file):
    for word in open(word_file):
        w = word.strip().split(",")
        if w[0]:
            yield unicode(w[0],"utf-8")

charweight = CharWeight(read_words("corpus.txt"))    
    
def memo(fn):
    cache = dict()
    def memoed(*args):
        if args not in cache:
            cache[args] = result = fn(*args)
            return result
        else:
            return cache[args]

    return memoed

# Edit distance functions

# Classic edit distance algorithm:
# en.wikipedia.org/wiki/Levenshtein_distance

@memo
def _edit_distance(s,t):
    """Return the minimum number of steps(delete,insert) to change s to t"""
    
    # insert char of t    
    if s == "":  return (sum([charweight(c) for c in t]),0)

    # delete char of s
    if t == "":  return (sum([charweight(c) for c in s]),0)

    if s[-1] == t[-1]:
        # match
        diff,match = _edit_distance(s[0:-1],t[0:-1])
        return diff, match + charweight(s[-1])
    else:
        # delete s's last char
        delete_diff,delete_match = _edit_distance(s[0:-1],t)
        # insert t's last char
        insert_diff,insert_match = _edit_distance(s,t[0:-1])

        if delete_diff < insert_diff:
            return delete_diff + charweight(s[-1]), delete_match
        else:
            return insert_diff + charweight(t[-1]), insert_match

def edit_distance(s,t):

    diff, same = _edit_distance(s,t)

    return (diff / (diff + same)) if (diff+same)!=0 else 0
    

def euclidean_distance(x,y):
    return sqrt(sum((x1-y1)**2 for x1,y1 in zip(x,y)))


class Cluster:

    def __init__(self, item, tag):
        """Initialize with one item"""
        self.items = [item]
        # This is used to select clustroid and calculate the radius of cluster
        self.sum_square_distance_to_other = [0]
        self.tag = item.tag = tag
        self.cache_clustroid = item
        self.cache_radius = 0


    def add(self,new_item, distance):
        """Add item to cluster, if item is assigned to new
        cluster(i.e. it's cluster tag is changed) return True,
        else return False"""

        # Invalidate cache
        self.cache_clustroid = self.cache_radius = None
        
        sum_square_distance_to_other = 0
        
        for i,x in enumerate(self.items):
            d = distance(new_item.item, x.item) ** 2
            sum_square_distance_to_other += d
            self.sum_square_distance_to_other[i] += d
        
        self.items.append(new_item)
        self.sum_square_distance_to_other.append(sum_square_distance_to_other)

        if new_item.tag != self.tag:
            new_item.tag = self.tag
            return True
        else:
            return False

    def clustroid(self):
        """Return a item represent whole cluster"""
        if self.cache_clustroid is None:
            self.calc_clustroid()
        return self.cache_clustroid

    def radius(self):
        # radius is the max distance between clustroid and other elements
        if self.cache_clustroid is None:
            self.calc_clustroid()
        return self.cache_radius

    def calc_clustroid(self):
        # Select the item whose distance to others is minimum
        self.cache_radius, self.cache_clustroid = min(zip(self.sum_square_distance_to_other,
                                                          self.items))
                                                          
    def get_items(self):
        return tuple(tag_item.item for tag_item in self.items)

    def get_tag(self):
        return self.tag

class Tag:
    """Wrap item with a tag slot"""
    
    def __init__(self,item,tag=None):
        self.item = item
        self.tag = tag

        
# K-means cluster algorithm
def k_means_cluster(items, k, distance, maxiter=10):

    def add_clusters(add_items, cluster_list):
        dbg("low","Add %d items to clusters" % len(add_items))
        tag_changed = False
        for tag_item in add_items:
            closest = min(cluster_list,
                          key = lambda c: distance(c.clustroid().item, tag_item.item))
            if closest.add(tag_item,distance): tag_changed = True
        return tag_changed

    def pick_k_items(to_be_picked):
        
        assert(len(to_be_picked) >= k)
        
        i = randint(0,len(to_be_picked)-1)
        k_items = []

        while True:
            
            k_items.append(to_be_picked[i])
            to_be_picked = to_be_picked[0:i] + to_be_picked[i+1:]

            if len(k_items) >= k: return k_items,to_be_picked

            # Pick the item which is far away from k_items
            _,i = max((min(distance(x.item,y.item) for y in k_items), j)
                       for j,x in enumerate(to_be_picked))

    # Wrap item with a cluster tag
    tag_items = [Tag(item) for item in items]

    if len(tag_items) <= k:
        return [Cluster(x,tag) for tag,x in enumerate(tag_items)]
            
    # Initialize k clusters by:
    # 1. randomly take one
    # 2. pick k-1 items apart from the first as far as possible
    
    k_items, to_be_assigned = pick_k_items(tag_items)

    clusters = [Cluster(x,tag) for tag,x in enumerate(k_items)]

    if dbg("low"):
        print_clusters(clusters)
        
    i = 0
    
    while True:
        
        # Assign the left items to k clusters
        changed = add_clusters(to_be_assigned,clusters)
        i += 1

        if dbg("low"):
            print_clusters(clusters)
        
        if not changed or i>=maxiter: break

        # With above results, re-initialize k clusters by choosing each cluster's clustroid,
        # and repeat the combine process, until no cluster change occurs

        clustroids = [x.clustroid() for x in clusters]

        clusters = [Cluster(x,x.tag) for x in clustroids]
        to_be_assigned = set(tag_items).difference(set(clustroids))

    if dbg("mid"):
        print "Iter %s times to compute k_means;Total radius:%s" % (i, sum([c.radius() for c in clusters]))
        
    return clusters


STOP_THRESHOLD = 0.01

def k_means(items,distance=edit_distance,maxiter=10):

    k = 1
    
    clusters = k_means_cluster(items,k,distance,maxiter)
    max_radius = max(c.radius() for c in clusters)

    while True:
        k += 1
        improved = k_means_cluster(items,k,distance,maxiter)
        improved_max_radius = max(c.radius() for c in improved)
        
        dbg("mid", "max_radius:%s, improved_max_radius: %s, improved ratio: %s",
            max_radius,improved_max_radius,
            (max_radius - improved_max_radius) / float(max_radius))
        
        if (max_radius - improved_max_radius) / float(max_radius) < STOP_THRESHOLD:
            return clusters
        max_radius = improved_max_radius
        clusters = improved
    
def print_clusters(clusters):

    for cluster in clusters:
        print
        print "<--- Cluster %s" % cluster.tag
        print "Items: %s, clustroid: %s, radius: %s" % (len(cluster.get_items()),
                                                        cluster.clustroid().item.encode("utf-8"),
                                                        cluster.radius())
        for word in sorted(cluster.get_items()):
            print word.encode("utf-8")
        print "---> Cluster %s" % cluster.tag            
            

def u(obj):

    if isinstance(obj,str):
        return unicode(obj,"utf-8")
    else:
        return obj
    
