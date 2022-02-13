import sys
import binascii
import math
from rich import print
from rich.console import Console
from rich.table import Column, Table
from rich.panel import Panel

verbose = False

console = Console()

stab_symbols = [
  bytes(0x00),
  bytes(0x01),
  bytes(0x02),
  bytes(0x03),
  bytes(0x04),
  bytes(0x02),
  bytes(0x02),
  bytes(0x02),
]

def log(str):
  if verbose:
    console.print(str)

def hex_to_bin(hex: bytes):
  return bin(int.from_bytes(hex, byteorder=sys.byteorder)) 

def bit_length(length: int) -> int:
  return (length - 1).bit_length()

def show_settings_info(settings):
  length = settings["length"]
  input_length = settings["input_length"]

  template = """
 * Max entiries: %d
 * Max index in bit length: %d
 * Input symbol size: %d
""" % (length, bit_length(length), input_length)

  print(Panel.fit(template, title="ASE Coding settings", border_style="blue", padding=0))

class LookupTable:
  def __init__(self, length: int, global_counter: int):
    self.entities = [None] * length
    self.length = length
    self.occupied = 0
    self.global_counter = global_counter
    self.culling_num = global_counter

  def push(self, symbol: int):
    log('[bold]Push symbol: [green]%s[/green][/bold]' % symbol)

    if symbol in self.entities[:self.occupied - 1]:
      hit_index = self.entities.index(symbol)
      log('[red] * Hit symbol on [bold]index %d[/bold] entry[/red]' % hit_index)
      self.arrange_table(symbol, hit_index)
      return hit_index

    log('[green] * New symbol[/green]')
    self.register_to_table(symbol)
    return -1
  
  def register_to_table(self, symbol: int):
    if self.occupied + 1 < self.length:
      self.entities.insert(0, symbol)
      self.occupied += 1

  def arrange_table(self, symbol: int, hit_index: int):
    for i in reversed(range(0, hit_index)):
      self.entities[i + 1] = self.entities[i]
    self.entities[0] = symbol
    self.entropy_culling()

  def entropy_culling(self):
    if self.global_counter > 0:
      self.global_counter -= 1
    else:
      log('[blue] * Invalidated the last entry[/blue]')
      if self.occupied > 0:
        self.occupied -= 1
      self.global_counter = self.culling_num

  def entropy_calc(self):
    return math.ceil(math.log2(self.occupied))

  def print_entries(self):
    print(' * Global counter: %d (Culling num: %d)' % (self.global_counter, self.culling_num))
    print(' * Occupied entries: %d' % self.occupied)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index \[bin]", style="dim", width=12)
    table.add_column("Entry")
    table.add_column("Label")

    for i in range(self.length):
      index = '%d [%s]' % (i, "{0:b}".format(i).zfill(bit_length(self.length)))
      label = 'Last' if i == self.occupied - 1 else 'Inv' if i > self.occupied - 1 and self.entities[i] != None else ''
      entry = bytes2binstr(self.entities[i]) if self.entities[i] != None else None
      table.add_row(index, entry, label)
      
    console.print(table)
    print('\n')

class ASECoding:
  def __init__(self, lookup_table):
    self.lookup_table = lookup_table

  def compress(self, symbol):
    hit_index = self.lookup_table.push(symbol)
    if hit_index == -1:
      cmark = 0
      return str(cmark) + bytes2binstr(symbol)

    cmark = 1
    m = self.lookup_table.entropy_calc()
    log("Entropy calc result %d" % m)
    return str(cmark) + "{0:b}".format(hit_index & ((1 << m) - 1))

def test_ase_compress_with_file(settings, filename):
  lt = LookupTable(settings["length"], settings["global_counter"])
  ase = ASECoding(lt)
  compressed = ''

  file = open(filename, 'rb')
  byte = file.read(1)
  a_comp_size = 0
  count = 0
  try:
    while byte:
      byte = file.read(1)
      a_comp_size += len(ase.compress(byte))
      count += 1
      if count % 100000 == 0:
        print(count)
  except Exception:
    print('Error')

  lt.print_entries()

  print('After compressed: %d bytes' % (a_comp_size / 8))

def test_ase_compress(settings, symbols):
  lt = LookupTable(settings["length"], settings["global_counter"])
  ase = ASECoding(lt)
  compressed = ''
  for symbol in symbols:
    compressed += ase.compress(symbol)
  lt.print_entries()

  b_comp_size = len(''.join(symbols))
  a_comp_size = len(compressed)

  print('Before compressed: %d bytes' % b_comp_size)
  print('After compressed: %d bytes (Compression rate: %d%%)' % (a_comp_size, (a_comp_size / b_comp_size) * 100))

def test_lookup_table_with_file(settings, filename):
  lt = LookupTable(settings["length"], settings["global_counter"])

  file = open(filename, 'rb')
  byte = file.read(1)
  count = 0

  while byte:
    byte = file.read(1)
    lt.push(byte)
    count += 1
    if count % 100000 == 0:
      print(count)

  lt.print_entries()

def test_lookup_table(settings, symbols):
  lt = LookupTable(settings["length"], settings["global_counter"])

  for symbol in symbols:
    lt.push(symbol)
  lt.print_entries()

def bytes2binstr(b, n=None):
  s = ' '.join(f'{x:08b}' for x in b)
  return s if n is None else s[:n + n // 8 + (0 if n % 8 else -1)]

ase_coding_settings = {
  "length": 8,         # Entry size
  "input_length": 8,   # Input size equals to 1 byte
  "global_counter": 4, # Global counter for entropy culling
}

filename = 'sky.bmp'

# test_lookup_table(ase_coding_settings, symbols=stab_symbols)
# test_lookup_table_with_file(ase_coding_settings, filename=filename)
# test_ase_compress(ase_coding_settings, symbols=stab_symbols)
test_ase_compress_with_file(ase_coding_settings, filename=filename)

show_settings_info(ase_coding_settings)