import random

class SparseGraph():
  def __init__(self, num_vertices, num_edges, seed=0):
    self.num_vertices = num_vertices
    self.num_edges = num_edges
    self.vertices = [i for i in range(num_vertices)]
    self.edges = [[] for i in range(num_vertices)]
    self.seed = seed
    self.random = random.Random(seed)
    self.nonzeros = []
    self.generate()
  
  def generate(self):
    for i in range(self.num_edges):
      src = self.random.randint(0, self.num_vertices - 1)
      dst = self.random.randint(0, self.num_vertices - 1)
      while(dst in self.edges[src] or src == dst):
        dst = self.random.randint(0, self.num_vertices - 1)
      self.edges[src].append(dst)
      random_float = self.random.uniform(0, 1)
      self.nonzeros.append(random_float)

  def convert_to_csr(self):
    values = []
    rows = []
    cols = []
    row_offset = 0
    rows.append(row_offset)
    for i in range(self.num_vertices):
      for j in self.edges[i]:
        cols.append(j)
        values.append(self.nonzeros[row_offset])
        row_offset += 1
      rows.append(row_offset)
    return values, rows, cols
    
  def convert_to_coo(self):
    values = []
    rows = []
    cols = []
    for i in range(self.num_vertices):
      for j in self.edges[i]:
        cols.append(j)
        rows.append(i)
    values = self.nonzeros
    return values, rows, cols
  
  def store_to_mtx_format(self, filename):
    values, rows, cols = self.convert_to_coo()
    with open(filename, 'w') as f:
      f.write('%%MatrixMarket matrix coordinate real general\n')
      f.write(f'{self.num_vertices} {self.num_vertices} {self.num_edges}\n')
      for i in range(self.num_edges):
        f.write(f'{rows[i]} {cols[i]} {values[i]}\n')

  def spmv(self, x):
    y = [0] * self.num_vertices
    non_zero_offset = 0
    for i in range(self.num_vertices):
      for j in self.edges[i]:
        y[i] += self.nonzeros[non_zero_offset] * x[j]
        non_zero_offset += 1
    return y