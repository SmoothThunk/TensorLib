/-
Copyright TensorLib Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/

import Batteries.Data.List -- for `toChunks`
import TensorLib.Broadcast
import TensorLib.Common
import TensorLib.Dtype
import TensorLib.Npy
import TensorLib.Shape

namespace TensorLib


/-!
A Tensor is the data bytes, along with all metadata required to do
efficient computation. The `startIndex`, `unitstrides` are inferred
from parsing or computed during a view creation.

Note that this representation is slightly different from the C version in NumPy source.

1. One difference is that we maintain "unit"-strides rather than strides. A unit stride
is just the stride divided by the datatype size. This makes indexing and iterating more
straightforward in my opinion. When you need to jump, you just need to remember to multiply
the number of slots by the datatype size.

2. Another is that we maintain a starting index into the array. Thus, if
we reverse a 1-D array, we keep the same ByteArray and update the start index. In C,
the `data` field is a pointer to a char array, and thus that can serve as the starting
point directly. This took me a while to figure out, so let me document an example

# x = np.arange(6, dtype='uint8')
# y = x[::-1]
# np.array_equal(y.base, x)
True

# x.ctypes.data
105553176936576

# y.ctypes.data
105553176936581

# y.ctypes.data - x.ctypes.data
5

Note that `x.data` and `y.data` exist, but are abstract types that, while there are addresses printed
with them, don't have this obvious behavior.

# y.base.data
<memory at 0x111694f40>

# x.data
<memory at 0x111694dc0>

y.base and x are the same so I don't know what the non-ctypes `data` field actually represents.
They certainly don't have the offset like the ctypes version.

If we decide to do reference counting and copying as in NumPy, we will
need this info, but for now we will copy whenever we update the array.

-/
-- TODO: Add a `base` field to track aliasing? NumPy does this and it may make sense for us.
-- TODO: Do we want this to be inductive to handle array scalars? https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars
--       Will force those into this type for now, but it seems wasteful.
--       NumPy has a bunch of special handling for array scalars.
-- TODO: I'm really not sure what to do with the data order here. Logically, the strides are
--       enough to navigate the tensor. E.g. x.T just reverses the strides, and everything works fine.
--       NumPy though, swaps the data order flag during a transpose as well. It is used for optimizations,
--       especially avoiding copies. We should get a good handle on this logic and either properly use
--       the data order field or remove it entirely.

structure Tensor where
  dtype : Dtype
  shape : Shape
  data : ByteArray
  startIndex : Nat := 0 -- Pointer to the first byte of ndarray data. This is implicit in the `data` pointer in numpy.
  unitStrides : Strides := shape.unitStrides
  deriving Repr, Inhabited, BEq

namespace Tensor

/-!
An tensor is trivially reshapable if it is contiguous with non-negative strides.
-/
def isTriviallyReshapable (arr : Tensor) : Bool :=
  arr.startIndex == 0
  && arr.unitStrides == arr.shape.unitStrides

def empty (dtype : Dtype) (shape : Shape) : Tensor :=
  let data := ByteArray.emptyWithCapacity (dtype.itemsize * shape.count)
  { dtype := dtype, shape := shape, data := data }

def zeros (dtype : Dtype) (shape : Shape) : Tensor := Id.run do
  let size := dtype.itemsize * shape.count
  let mut data := ByteArray.emptyWithCapacity size
  for _ in [0:size] do
    data := data.push 0
  { dtype := dtype, shape := shape, data := data }

--! number of dimensions
def ndim (x : Tensor) : Nat := x.shape.ndim

--! number of elements
def size (x : Tensor) : Nat := x.shape.count

--! number of bytes representing each element
def itemsize (x : Tensor) : Nat := x.dtype.itemsize

--! Get the offset corresponding to a DimIndex
def dimIndexToOffset (x : Tensor) (i : DimIndex) : Int :=
  Shape.dimIndexToOffset x.unitStrides i

--! Get the starting byte corresponding to a DimIndex
def dimIndexToPosition (x : Tensor) (i : DimIndex) : Nat :=
  (x.startIndex + (x.itemsize * x.dimIndexToOffset i)).toNat

--! number of bytes representing the entire tensor
def nbytes (x : Tensor) : Nat := x.itemsize * x.size

def isIntLike (x : Tensor) : Bool := x.dtype.isIntLike

def dimIndexInRange (arr : Tensor) (dimIndex : DimIndex) : Bool := arr.shape.dimIndexInRange dimIndex

def byteArrayAtDimIndex (arr : Tensor) (dimIndex : DimIndex) : Err ByteArray := do
  if !arr.dimIndexInRange dimIndex then .error "index is incompatible with tensor shape" else
  let posn := arr.dimIndexToPosition dimIndex
  .ok $ arr.data.extract posn (posn + arr.itemsize)

def setByteArrayAtDimIndex (arr : Tensor) (dimIndex : DimIndex) (bytes : ByteArray) : Err Tensor := do
  if !arr.dimIndexInRange dimIndex then .error "index is incompatible with tensor shape" else
  if arr.itemsize != bytes.size then .error "byte size mismatch" else
  let posn := arr.dimIndexToPosition dimIndex
  .ok $ { arr with data := bytes.copySlice 0 arr.data posn bytes.size }

/-!
Return the integer at the dimIndex. This is useful, for example, in advanced indexing
where we must have an int/uint Tensor as an argument.
-/
def intAtDimIndex (arr : Tensor) (dimIndex : DimIndex) : Err Int := do
  if !arr.isIntLike then .error "natAt expects an int tensor" else
  let bytes <- byteArrayAtDimIndex arr dimIndex
  .ok $ bytes.toInt

/-!
Copy a Tensor's data to new, contiguous storage.

Note that this doesn't just do a `ByteArray` copy. It walks over `arr` according
to its strides and selects the elements, so it works on views of other tensors with
non-contiguous data.
-/
def copy (arr : Tensor) : Tensor := Id.run do
  let itemsize := arr.dtype.itemsize
  let mut data := ByteArray.emptyWithCapacity arr.nbytes
  for dimIndex in arr.shape.belist do
    let posn := arr.dimIndexToPosition dimIndex
    for j in [0:itemsize] do
      let b := arr.data.get! (posn + j)
      -- Leaving this here for posterity. Random access `set!` doesn't work on
      -- arrays created using `ByteArray.empty`, even if the initial size is greater
      -- than the argument to `set!`. It just quietly does nothing.
      -- data := data.set! i b
      data := data.push b
  let arr := {
     dtype := arr.dtype,
     shape := arr.shape,
     data
  }
  return arr

/-
Reshaping is surprisingly tricky. These are the main methods in NumPy.

  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L187-L189
  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L195-L197
  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L347
    let olds := x.shape

It's a couple hundred lines of C code with many corner cases. For example, it gets
hard when the array is already a view, especially one that is already non-contiguous.
If the array is contiguous, and strides are positive, things are much easier;
we can just write down the new shape and compute the new strides. Things that complicate
the picture, and require all the logic in the code above:

1. Axes of size 1 have a non-0 stride, but it shouldn't be used in new stride calculations.
2. If the array is not contiguous, e.g. created with a non-unit `step` in a slice, we
   may or may not be able to represent it without copying.
3. Even if the array is contiguous, with negative strides, e.g. created with a negative `step` in a slice, we
   may or may not be able to represent it without copying. For example, if some strides are positive and
   some are negative, it can require a copy to reshape.
4. Data ordering (C vs Fortran, row major vs column major) goes into the calculation of whether an array
   is contiguous or not. The same strides on the same data can be contiguous or not depending on the data ordering.

Examples:

Reshapes can require a copy. For example, when we get the data out of order via
reverses and reshapes, flattening it again will require a copy.

# x = np.arange(6)

# x.reshape(3, 2).base is x
True

# x.reshape(3, 2)[::-1].base is x
True

# x.reshape(3, 2)[::-1].reshape(6)
array([4, 5, 2, 3, 0, 1])

# x.reshape(3, 2)[::-1].reshape(6).base is x
False

Clearly a simple list of numbers for strides can't jump around the original list
to capture that pattern. My guess is that there is we can figure out if we need
a copy by looking at startPosition, shape, and strides.
-/
private def copyAndReshape (arr : Tensor) (shape : Shape) : Err Tensor :=
  if arr.shape.count != shape.count then .error s!"Incompatible shapes: {arr.shape} {shape}" else
  let arr := arr.copy
  .ok { arr with shape, unitStrides := shape.unitStrides }

def copyAndReshape! (arr : Tensor) (shape : Shape) : Tensor :=
  get! (copyAndReshape arr shape)

def reshape (arr : Tensor) (shape : Shape) : Err Tensor :=
  if arr.shape == shape then .ok arr else
  if arr.shape.count != shape.count then .error s!"Incompatible shapes: {arr.shape} {shape}" else
  if arr.isTriviallyReshapable
  then .ok { arr with shape, unitStrides := shape.unitStrides }
  else copyAndReshape arr shape

def reshape! (t : Tensor) (s : Shape) : Tensor := get! $ t.reshape s

/-
NumPy allows you to transpose flexibly on the axes, e.g. allowing arbitrary
reorders. This is just the default, which reverses the dimensions.
It is also extremely conservative when it comes to needing copies.
-/
def transpose (arr : Tensor) : Tensor :=
  let shape : Shape := arr.shape.map List.reverse
  if arr.isTriviallyReshapable
  then { arr with shape, unitStrides := shape.unitStrides }
  else
    let arr := arr.copy
    { arr with shape, unitStrides := shape.unitStrides }

-- The result shape is always equal to `toShape` so we don't need to remember it
private def broadcastStrides (fromShapeAndStrides : List (Nat × Int)) (toShape : Shape) : Err Strides :=
  let rec loop (fromShapeAndStrides : List (Nat × Int)) (toShape : List Nat) : Err Strides :=
    match fromShapeAndStrides, toShape with
    | [], [] => .ok []
    | (xDim, xStride) :: xs, yDim :: ys =>
      if xDim == yDim then do
        let rest <- loop xs ys
        return xStride :: rest
      else if xDim == 1 then do
        let rest <- loop xs ys
        return 0 :: rest
      else .error "Can't broadcast dimension"
    | [], _ :: ys => do
      let rest <- loop [] ys
      return 0 :: rest
    | _ :: _, [] => .error "Can't broadcast dimension"
  (loop fromShapeAndStrides.reverse toShape.val.reverse).map fun x => x.reverse

#guard broadcastStrides [] (Shape.mk []) == .ok []
#guard broadcastStrides [(5, 8)] (Shape.mk [5]) == .ok [8]
#guard broadcastStrides [(5, 8)] (Shape.mk [10, 5]) == .ok [0, 8]
#guard broadcastStrides [(5, 8)] (Shape.mk [15, 10, 5]) == .ok [0, 0, 8]
#guard !(broadcastStrides [(1, 0)] (Shape.mk [])).isOk
#guard !(broadcastStrides [(1, 0), (2, 2), (3, 8)] (Shape.mk [2, 3])).isOk
#guard !(broadcastStrides [(1, 0)] (Shape.mk [])).isOk
#guard !(broadcastStrides [(2, 0), (2, 2), (3, 8)] (Shape.mk [1, 2, 3])).isOk

/-
Theorems to prove:
* arr.broadcastTo arr.shape == arr
* (arr.broadcastTo s1).broadcastTo s2 == arr.broadcastTo s2
* ...
-/
def broadcastTo (arr : Tensor) (shape : Shape) : Err Tensor :=
  match Broadcast.broadcast { left := arr.shape, right := shape } with
  | none => .error s!"Can't broadcast {arr.shape} to {shape}"
  | some shape' =>
    if shape != shape' then .error s!"Can't broadcast {arr.shape} to {shape}" else do
    let strides <- broadcastStrides (arr.shape.val.zip arr.unitStrides) shape
    .ok $ Tensor.mk arr.dtype shape arr.data arr.startIndex strides

def broadcastTo! (arr : Tensor) (shape : Shape) : Tensor := get! $ broadcastTo arr shape

def broadcast (arr1 : Tensor) (arr2 : Tensor) : Err (Tensor × Tensor) :=
  match Broadcast.broadcast { left := arr1.shape, right := arr2.shape } with
  | none => .error "Can't broadcast"
  | some shape => do
  let arr1 <- arr1.broadcastTo shape
  let arr2 <- arr2.broadcastTo shape
  return (arr1, arr2)

def arrayScalar (dtype : Dtype) (arr : ByteArray) : Err Tensor :=
  if dtype.itemsize != arr.size then .error s!"data size mismatch: {dtype} {arr.size}" else
  .ok { dtype := dtype, shape := Shape.empty, data := arr }

def arrayScalar! (dtype : Dtype) (arr : ByteArray) : Tensor := get! $ arrayScalar dtype arr

def arrayScalarNat (dtype : Dtype) (n : Nat) : Err Tensor := do
  let arr <- dtype.byteArrayOfNat n
  arrayScalar dtype arr

def arrayScalarNat! (dtype : Dtype) (n : Nat) : Tensor := get! $ arrayScalarNat dtype n

def arrayScalarInt (dtype : Dtype) (n : Int) : Err Tensor := do
  let arr <- dtype.byteArrayOfInt n
  arrayScalar dtype arr

def arrayScalarInt! (dtype : Dtype) (n : Int) : Tensor := get! $ arrayScalarInt dtype n

def arrayScalarBool  (b : Bool) : Err Tensor := arrayScalar Dtype.bool (toLEByteArray b)

def arrayScalarBool! (b : Bool) : Tensor := get! $ arrayScalarBool b

def arrayScalarFloat32 (f : Float32) : Err Tensor := arrayScalar Dtype.float32 (toLEByteArray f)

def arrayScalarFloat32! (f : Float32) : Tensor := get! $ arrayScalarFloat32 f

def arrayScalarFloat64 (f : Float) : Err Tensor := arrayScalar Dtype.float64 (toLEByteArray f)

def arrayScalarFloat64! (n : Float) : Tensor := get! $ arrayScalarFloat64 n

def arange (dtype : Dtype) (n : Nat) : Err Tensor := do
  let size := dtype.itemsize
  let mut data := ByteArray.emptyWithCapacity (n * size)
  for i in [0:n] do
    let bytes <- dtype.byteArrayOfNat i
    data := ByteArray.copySlice bytes 0 data (i * size) size
  return { dtype := dtype, shape := Shape.mk [n], data }

def arange! (dtype : Dtype) (n : Nat) : Tensor := get! $ arange dtype n

-- This is a blind index into the array, disregarding the shape.
def getPosition (arr : Tensor) (position : Nat) : ByteArray :=
  arr.data.copySlice (position * arr.itemsize) (ByteArray.emptyWithCapacity arr.itemsize) 0 arr.itemsize

#guard
  let tp :=  Dtype.uint32
  let arr := arange! tp 4
  let v := getPosition arr 3
  v.toList == [3, 0, 0, 0]

-- This is a blind index into the array, disregarding the shape.
def setPosition (arr : Tensor) (n : Nat) (v : ByteArray): Tensor :=
  let size := arr.itemsize
  let posn := n * size
  { arr with data := v.copySlice 0 arr.data posn size }

#guard
  let tp :=  Dtype.uint8
  let arr := arange! tp 4
  let arr := setPosition arr 0 (tp.byteArrayOfNat! 7)
  arr.data.data == #[7, 1, 2, 3]

/-
We now define some constructors for Tensors that are mostly useful
for testing. Instead of requiring the `dtype` argument, it may be
better to use a class, e.g.

    class DtypeOf a where
      dtype : Dtype

and let `ofList` infer the dtype. The reason we're not going with
this now is that there is no obvious canonical candidate for the
dtype. E.g. for Nat, we could reasonably want uint8 for small examples,
uint16 for bigger ones, etc.
-/
def ofNatList (dtype : Dtype) (ns : List Nat) : Err Tensor := do
  if !dtype.isUint then .error "not a uint type" else
  let size := dtype.itemsize
  let arr := Tensor.zeros dtype (Shape.mk [ns.length])
  let mut data := arr.data
  let mut posn := 0
  for n in ns do
    let v <- dtype.byteArrayOfNat n
    data := v.copySlice 0 data posn size
    posn := posn + size
  .ok { arr with data := data }

def ofNatList! (dtype : Dtype) (ns : List Nat) : Tensor := get! $ ofNatList dtype ns

def ofIntList (dtype : Dtype) (ns : List Int) : Err Tensor := do
  if !dtype.isInt then .error "not an int type" else
  let size := dtype.itemsize
  let arr := Tensor.zeros dtype (Shape.mk [ns.length])
  let mut data := arr.data
  let mut posn := 0
  for n in ns do
    let v <- dtype.byteArrayOfInt n
    data := v.copySlice 0 data posn size
    posn := posn + size
  .ok { arr with data := data }

def ofIntList! (dtype : Dtype) (ns : List Int) : Tensor := get! $ ofIntList dtype ns

def ofFloat32List (ns : List Float32) : Err Tensor := do
  let dtype := TensorLib.Dtype.float32
  let size := dtype.itemsize
  let arr := Tensor.zeros dtype (Shape.mk [ns.length])
  let mut data := arr.data
  let mut posn := 0
  for n in ns do
    let v <- dtype.byteArrayOfFloat32 n
    data := v.copySlice 0 data posn size
    posn := posn + size
  .ok { arr with data := data }

def ofFloat32List! (ns : List Float32) : Tensor := get! $ ofFloat32List ns

def getDimIndex (arr : Tensor) (index : DimIndex) : Err ByteArray :=
  if arr.shape.ndim != index.length then .error "getDimIndex: index mismatch" else
  let offset := Shape.dimIndexToOffset arr.unitStrides index
  let posn := arr.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}" else
  let res := getPosition arr posn.toNat
  .ok res

def getDimIndex! (arr : Tensor) (index : DimIndex) : ByteArray := get! $ getDimIndex arr index

#guard
  let arr := arrayScalarNat! Dtype.uint8 25
  let v := getDimIndex! arr []
  v.data == #[25]

def setDimIndex (arr : Tensor) (index : DimIndex) (v : ByteArray) : Err Tensor :=
  let offset := Shape.dimIndexToOffset arr.unitStrides index
  let posn := arr.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else .ok $ setPosition arr posn.toNat v

def toList (arr : Tensor) : Err (List ByteArray) :=
  arr.shape.allDimIndices.mapM (getDimIndex arr)

/-
Similar to np.array_equal, but requires the dtype to be the same
-/
def arrayEqual (x y : Tensor) : Bool :=
  x.dtype == y.dtype && x.shape == y.shape && match x.toList, y.toList with
  | .error _, _ | _, .error _ => false
  | .ok xs, .ok ys => xs.length == ys.length && (xs.zip ys).all fun (x, y) => x == y

/-
Like NumPy's `astype`: https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.astype.html
Afaict, NumPy never fails due to overflow/underflow during type conversions, so use the "overflow"
variant of type casting.

`astype` will make a copy of the tensor iff `toDtype != arr.dtype`.
-/
def astype (arr : Tensor) (toDtype : Dtype) : Err Tensor := do
  if arr.dtype == toDtype then .ok arr else
  let mut res : Tensor := {
    dtype := toDtype,
    shape := arr.shape,
    data := ByteArray.emptyWithCapacity (arr.size * toDtype.itemsize)
  }
  for dimIndex in arr.shape.belist do
    let v <- arr.getDimIndex dimIndex
    let v' <- Dtype.castOverflow arr.dtype v toDtype
    let res' <- res.setDimIndex dimIndex v'
    res := res'
  return res

def astype! (arr : Tensor) (toDtype : Dtype) : Tensor := get! $ astype arr toDtype

def asFloat (arr : Tensor) : Err Tensor := arr.astype arr.dtype.floatVariant

def ofBoolList (dtype : Dtype) (ns : List Bool) : Err Tensor := do
  let t <- ofIntList dtype $ ns.map fun n => if n then 1 else 0
  t.astype Dtype.bool

def ofBoolList! (dtype : Dtype) (ns : List Bool) : Tensor := get! $ ofBoolList dtype ns

namespace Format
open Std.Format

-- Useful for small arrays, e.g. to help with printing and such
-- There are some natural invariants we could check, such as that the
-- trees in a node all have the same height, but since this is just a
-- utility structure we'll keep it simple
inductive Tree a where
| root (xs: List a)
| node (xs: List (Tree a))
deriving BEq, Repr, Inhabited

namespace Tree

-- Traverse the left-most branch to infer the shape. Don't bother checking that it's uniform
-- since presumably it was created by a `arr.toTree` variant.
private def inferShape (t : Tree a) : List Nat := match t with
| .root xs => [xs.length]
| .node [] => impossible
| .node (t :: ts) => (1 + ts.length) :: inferShape t

def mapM [Monad m] (f : a -> m b) (t : Tree a) : m (Tree b) :=
  map1 f t
where
  map1 f
  | .root xs => do
    let xs' <- xs.mapM f
    return .root xs'
  | .node ts => do
    let ts' <- mapN f ts
    return .node ts'
  mapN f
  | [] => return []
  | t :: ts => do
    let t' <- map1 f t
    let ts' <- mapN f ts
    return t' :: ts'

def map (f : a -> Id b) (t : Tree a) : Tree b := mapM f t

private def formatRoot [Repr a] (xs : List a) : Lean.Format :=
  sbracket (joinSep (List.map repr xs) (text ", "))

private def formatTree1 [Repr a] (t : Tree a) (shape : List Nat) : Err Std.Format :=
  match shape, t with
  | [], .root [x] => .ok $ repr x
  | [n], .root r => if r.length != n then .error "shape mismatch" else .ok (formatRoot r)
  | n :: shape, .node ts => do
    let fmts <- ts.traverse (fun t => formatTree1 t shape)
    if fmts.length != n then .error "head mismatch" else
    let indented := join (fmts.intersperse (", " ++ line))
    .ok (group (nest 2 ("[" ++ indented ++ "]")))
  | _, _ => .error "format mismatch"

def format [Repr a] (t : Tree a) : Err Lean.Format := do
  let r <- formatTree1 t t.inferShape
  return join ["array(", r, ")"]

def format! [Repr a] (t : Tree a) : Lean.Format := get! $ format t

end Tree

private def listToTree (arr : List a) (strides : Strides) : Err (Tree a) :=
  if strides.any fun x => x <= 0 then .error "strides need to be positive" else
  match strides with
  | [] => if arr.length == 1 then .ok (.root arr) else .error "empty shape that's not an array scalar"
  | [1] => .ok (.root arr)
  | [_] => .error "not a unit stride"
  | stride :: strides => do
    let chunks := arr.toChunks stride.toNat
    let res <- chunks.mapM (fun x => listToTree x strides)
    return .node res


/- This needs some improvement. For example, I'm not able to get the indent to stick
at the end of the "array("

$ bin/tensorlib format 20 2
Got shape [20, 2]
array([[0x0000#16, 0x0001#16],
  [0x0002#16, 0x0003#16],
  [0x0004#16, 0x0005#16],
  ...
-/

end Format

def toByteArrayTree (arr : Tensor) : Err (Format.Tree ByteArray) := do
  let xs <- arr.toList
  -- Now that we have the elements in a list, we don't care about the strides `arr` which
  -- could have been complex (e.g. negative). Now we just want standard unit strides over the list
  Format.listToTree xs arr.shape.unitStrides

def toIntTree (arr : Tensor) : Err (Format.Tree Int) := do
  let t <- arr.toByteArrayTree
  return t.map ByteArray.toInt

def toIntTree! (arr : Tensor) : Format.Tree Int := get! $ toIntTree arr

def toNatTree (arr : Tensor) : Err (Format.Tree Nat) := do
  let t <- arr.toByteArrayTree
  return t.map ByteArray.toNat

def toNatTree! (arr : Tensor) : Format.Tree Nat := get! $ toNatTree arr

-- decode each element to fp32
-- Errors propagate instead of being silently replaced with 0 using mapM instead of map.
def toFloat32Tree (arr : Tensor) : Err (Format.Tree Float32) := do
  let t <- arr.toByteArrayTree
  match arr.dtype with
  | .float8_e2m5 => t.mapM (fun b => Dtype.decodeFloat8E2M5 b)
  | .float8_e5m2 => t.mapM (fun b => Dtype.decodeFloat8E5M2 b)
  | .float8_e4m3 => t.mapM (fun b => Dtype.decodeFloat8E4M3 b)
  | .float8_e3m4 => t.mapM (fun b => Dtype.decodeFloat8E3M4 b)
  -- e8m0 is a scale-only type, but we include a decode case here to prevent
  -- the default branch from misinterpreting 1-byte e8m0 data as multi-byte fp32/fp64.
  | .float8_e8m0 => t.mapM (fun b => Dtype.decodeFloat8E8M0 b)
  | .float16 => t.mapM (fun b => Dtype.byteArrayToFloat16 .float16 b)
  | .bfloat16 => t.mapM (fun b => Dtype.byteArrayToBFloat16 .bfloat16 b)
  | _ => t.mapM ( fun b => Float32.ofLEByteArray b)

def toFloat32Tree! (arr : Tensor) : Format.Tree Float32 := get! $ toFloat32Tree arr

def toFloat64Tree (arr : Tensor) : Err (Format.Tree Float) := do
  let t <- arr.toByteArrayTree
  match arr.dtype with
  | .float8_e2m5 => t.mapM (fun b => do let f <- Dtype.decodeFloat8E2M5 b; return f.toFloat)
  | .float8_e5m2 => t.mapM (fun b => do let f <- Dtype.decodeFloat8E5M2 b; return f.toFloat)
  | .float8_e4m3 => t.mapM (fun b => do let f <- Dtype.decodeFloat8E4M3 b; return f.toFloat)
  | .float8_e3m4 => t.mapM (fun b => do let f <- Dtype.decodeFloat8E3M4 b; return f.toFloat)
  | .float8_e8m0 => t.mapM (fun b => do let f <- Dtype.decodeFloat8E8M0 b; return f.toFloat)
  | .float16 => t.mapM (fun b => do let f <- Dtype.byteArrayToFloat16 .float16 b; return f.toFloat)
  | .bfloat16 => t.mapM (fun b => do let f <- Dtype.byteArrayToBFloat16 .bfloat16 b; return f.toFloat)
  | .float32 => t.mapM (fun b => do let f <- Float32.ofLEByteArray b; return f.toFloat)
  | _ =>  t.mapM (fun b => Float.ofLEByteArray b)


def toFloat64Tree! (arr : Tensor) : Format.Tree Float := get! $ toFloat64Tree arr

def toBoolTree (arr : Tensor) : Err (Format.Tree Bool) := do
  let t <- arr.toByteArrayTree
  return t.map ByteArray.toBool

def toBoolTree! (arr : Tensor) : Format.Tree Bool := get! $ toBoolTree arr

def formatInt (arr : Tensor) : Err Std.Format := do
  let t <- arr.toIntTree
  t.format

def formatNat (arr : Tensor) : Err Std.Format := do
  let t <- arr.toNatTree
  t.format

private def reverseEndianness (arr : ByteArray) (itemsize : Nat) : Err ByteArray := do
  if arr.size.mod itemsize != 0 then .error "Bytearray size mismatch" else
  let mut res := ByteArray.emptyWithCapacity arr.size
  for i in [0:arr.size / itemsize] do
    let offset := itemsize * i
    let bytes := arr.extract offset (offset + itemsize)
    let bytes := bytes.reverse
    res := res.append bytes
  return res

private def dataOfNpy (arr : Npy.Ndarray) : Err ByteArray := do
  let dst := ByteArray.emptyWithCapacity arr.nbytes
  let copied := arr.data.copySlice arr.startIndex dst 0 arr.nbytes
  let res <- match arr.order with
  | .notApplicable
  | .littleEndian => .ok copied
  | .bigEndian => reverseEndianness copied arr.itemsize
  | .native => .error "Native byte ordering is not supported. Please force a byte order when you save the array."
  return res

/-
Makes a copy of the data, dropping the header and padding.
Probably not a great choice, but sticking with it for now.
I want to avoid writing .npy files with wrong header data.
-/
def ofNpy (arr : Npy.Ndarray) : Err Tensor := do
  let dtype := arr.dtype.name
  let shape := arr.header.shape
  let data <- dataOfNpy arr
  let startIndex := 0
  return { dtype, shape, data, startIndex }

/-
If we have a non-trivial view, we will need a copy, since strides
and start positions are not included in the .npy file format
-/
def toNpy (arr : Tensor) : Err Npy.Ndarray :=
  -- fp8_e3m4 and fp8_e4m3 both serialize to "<V1" in the npy header with no distinguishing
  -- metadata. Ml_dtypes has this limitation too: np.save followed by np.load returns <V1 bytes,
  -- losing the original fp8 type. This guard protects library users from a silent round-trip
  -- failure: without it, saving an e3m4 tensor and loading it back would interpret the bytes
  -- as e4m3 (wrong values, no error).
  -- Note: tests create e3m4 .npy files via Python's np.save and decode the bytes directly
  -- with decodeFloat8E3M4 - they don't use this function.
  -- Our guard helps to surface an explicit error during write instead of allowing a
  -- silent round-trip corruption — without it, a user could save an e3m4 tensor, load it back,
  -- and get wrong values (interpreted as e4m3) with no indication anything went wrong.
  if arr.dtype == .float8_e3m4 || arr.dtype == .float8_e2m5 || arr.dtype == .float8_e8m0 then .error "float8_e3m4/float8_e2m5/float8_e8m0 cannot be saved to npy: format uses V1 which is indistinguishable from float8_e4m3"
  else
    let arr := if arr.isTriviallyReshapable then arr else arr.copy
    let descr := Npy.Dtype.mk arr.dtype Npy.ByteOrder.littleEndian
    let shape := arr.shape
    let header : Npy.Header := { descr := descr, shape := shape }
    let data := arr.data
    let startIndex := 0
    .ok { header, data, startIndex }

-- Dequantize an MX scaled tensor: v_i = decodeE8M0(scale) x fp32(qW_i), one scale per group
-- reconstructs the original fp32 values from a block scaled quantized tensor by multiplying each elemt by its group's decoded E8M0 scale
def dequantizeMX (qW : Tensor) (scales : Tensor) (groupSize: Nat) : Err Tensor := do
  -- scales tensor must be E8M0 (the only MX scale format supported by tensorlib)
  if qW.dtype != .float32 then .error "dequantizeMX: qW must have dtype float32"
  else if scales.dtype != .float8_e8m0 then .error "dequantizeMX: scales must have dtype float8_e8m0"
  else
  -- dequantization is defined along the last dimension so we need atleast 1
  let lastDim <- match qW.shape.val.getLast? with
    | none => .error "dequantizeMX: qW must have atleast one dimension"
    | some d => .ok d
  -- each group of groupSize elements shares one scale byte
  if groupSize == 0 then .error "dequantizeMX: groupSize must be positive"
  else if lastDim % groupSize != 0 then
    .error "dequantizeMX: groupSize must divide the last dimension of qW"
  else
    -- expected scales shape same as qW but last dimension is divided by groupSize
    let expectedScalesShape := TensorLib.Shape.mk (qW.shape.val.dropLast ++ [lastDim / groupSize])
    if scales.shape != expectedScalesShape then
      .error "dequantizeMX: Scales shape does not match qW shape / groupSize"
    else
      -- flatten both tensors to lists of raw bytes, one byteArray per element
      let qWElems <- qW.toList
      let scElems <- scales.toList
      -- split qW elements into consecutive groups of groupSize
      -- each group corresponds to one scale byte in scElems
      let groups := List.toChunks groupSize qWElems
      -- zip each group of qW elements with its corresponding scale byte
      -- for each pair: decode scale, decode each element, multiply (v_i = X * P_i)
      let resultGroups <- (groups.zip scElems).mapM fun (group, scaleBytes) => do
        -- decode E8M0 scale byte to fp32: X = 2^(byte - 127), 0xFF -> NaN
        let X <- Dtype.decodeFloat8E8M0 scaleBytes
        -- for each element in the group, decode to fp32 and multiply by scale
        group.mapM fun elemBytes => do
          let p <- Dtype.byteArrayToFloat32 qW.dtype elemBytes
          -- OCP MX spec §5.1: v_i = X * P_i
          let v := X * p
          -- pack result back as fp32 bytes
          Dtype.byteArrayOfFloat32 .float32 v
      -- flatten result groups back to a single list of byte array
      let flatElems := resultGroups.flatten
      -- concatenate all bytes into a single byte array
      let data := flatElems.foldl (fun acc bytes => acc.append bytes) (ByteArray.emptyWithCapacity (flatElems.length * Dtype.float32.itemsize))
      -- fp32 tensor with same shape as qW
      return {dtype := .float32, shape := qW.shape, data := data}

-- guards for tricky cases for dequantize
-- case 1: fractional scale (byte 126 = 0.5), groupSize 1
-- 3.0 * 0.5 = 1.5, 5.0 * 0.5 = 2.5
#guard
  let qW := Tensor.ofFloat32List! [3.0, 5.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [2], data := ByteArray.mk #[126, 126] : Tensor }
  match Tensor.dequantizeMX qW scales 1 with
  | .error _ => false
  | .ok result => result.toFloat32Tree! == .root [1.5, 2.5]

-- case 2: different scales per group
-- group 1: byte 128 = 2.0, so [2.0, 4.0] -> [4.0, 8.0]
-- group 2: byte 126 = 0.5, so [6.0, 8.0] -> [3.0, 4.0]
#guard
  let qW := Tensor.ofFloat32List! [2.0, 4.0, 6.0, 8.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [2], data := ByteArray.mk #[128, 126] : Tensor }
  match Tensor.dequantizeMX qW scales 2 with
  | .error _ => false
  | .ok result => result.toFloat32Tree! == .root [4.0, 8.0, 3.0, 4.0]

-- case 3: negative values, sign must be preserved
-- byte 128 = 2.0, so [-2.0, -4.0] -> [-4.0, -8.0]
#guard
  let qW := Tensor.ofFloat32List! [-2.0, -4.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [1], data := ByteArray.mk #[128] : Tensor }
  match Tensor.dequantizeMX qW scales 2 with
  | .error _ => false
  | .ok result => result.toFloat32Tree! == .root [-4.0, -8.0]

-- case 4: scale byte 0 = 2^-127 (smallest E8M0, fp32 subnormal)
-- 1.0 * 2^-127 = fp32 subnormal 0x00400000
#guard
  let qW := Tensor.ofFloat32List! [1.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [1], data := ByteArray.mk #[0] : Tensor }
  match Tensor.dequantizeMX qW scales 1 with
  | .error _ => false
  | .ok result => result.toFloat32Tree! == .root [Float32.ofBits 0x00400000]

-- case 5: scale byte 254 = 2^127 (largest E8M0 value)
-- 1.0 * 2^127 = Float32.ofBits 0x7F000000
#guard
  let qW := Tensor.ofFloat32List! [1.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [1], data := ByteArray.mk #[254] : Tensor }
  match Tensor.dequantizeMX qW scales 1 with
  | .error _ => false
  | .ok result => result.toFloat32Tree! == .root [Float32.ofBits 0x7F000000]

-- case 6: mixed NaN and non-NaN groups
-- group 1: byte 127 = 1.0, so [2.0, 4.0] -> [2.0, 4.0]
-- group 2: byte 255 = NaN, so [6.0, 8.0] -> [NaN, NaN] per OCP 5.1
#guard
  let qW := Tensor.ofFloat32List! [2.0, 4.0, 6.0, 8.0]
  let scales := { dtype := .float8_e8m0, shape := TensorLib.Shape.mk [2], data := ByteArray.mk #[127, 255] : Tensor }
  match Tensor.dequantizeMX qW scales 2 with
  | .error _ => false
  | .ok result => match result.toFloat32Tree! with
    | .root [a, b, c, d] => a == 2.0 && b == 4.0 && c.isNaN && d.isNaN
    | _ => false

-- Quantize a fp32 tensor to MX format using NVIDIA's scale computation:
-- m = floor_pw2(fp8Max / amax), scale byte = floor(log2(1/m)) + 127
-- Reference: NVIDIA TensorEngine
-- Returns (qW, scales) where qW is fp32 scaled values and scales is e8m0 byte tensor
def quantizeMX (x : Tensor) (groupSize : Nat) (computeDtype : Dtype) : Err (Tensor × Tensor) := do
  -- lookup fp8Max for the compute dtype -- returns none for non fp8 dtypes
  let fp8Max <- match Dtype.fp8Max computeDtype with
    | none => .error s!"quantizeMX: unsupported compute dtype {computeDtype}"
    | some v => .ok v
  -- x must be fp32
  if x.dtype != .float32 then .error "quantizeMX: inpute tensor must be float32"
  else
    -- last dim must divide evenly by groupsize
    let lastDim <- match x.shape.val.getLast? with
      | none => .error "quantizeMX: input tensor must have >= 1 dimension"
      | some d => .ok d
    if groupSize == 0 then .error "quantizeMX: groupSize must be positive"
    else if lastDim % groupSize != 0 then .error "quantizeMX: groupSize must divide the last dimension of x" else
    -- flatten x to a list of raw bytes, one ByteArray per element
    let xElems <- x.toList
    -- split into consecutive groups of groupSize along the last dim
    let groups := List.toChunks groupSize xElems
    -- for each group, compute the E8M0 scale byte using NVIDIA's formula:
    -- m = floor_pow2(fp8Max / amax), scale byte = floor(log2(1/m)) + 127
    let results <- groups.mapM fun group => do
      -- decode each element to Float32
      let vals <- group.mapM (Dtype.byteArrayToFloat32 .float32)
      -- amax = max absolute value in the group
      let amax := vals.foldl (fun acc v => if v.abs > acc then v.abs else acc) 0.0
      -- compute scale byte and multiplier together (avoids recomputing ratio/logM)
      let ratio := fp8Max / amax
      let (scaleByte, m) : UInt8 × Float32 :=
        if amax == 0.0 then (127, 1.0)
        else if amax.isInf || amax.isNaN then (255, 1.0)
        else if ratio.isInf then (254, Float32.ofBits 0x7F000000)
        else
          let logM := ratio.log2.floor
          let s := (-logM + 127.0)
          let byte := if s < 0.0 then 0
                      else if s > 254.0 then 254
                      else s.toUInt8
          (byte, Float32.pow 2.0 logM)
        let scaledVals <- vals.mapM fun v => Dtype.byteArrayOfFloat32 .float32 (v * m)
      return (scaledVals, scaleByte)
    -- separate scaled values and bytes from results
    let scaledGroups := results.map Prod.fst
    let scaleBytes := results.map Prod.snd
    -- flatten scaled groups into a single ByteArray for qW
    let flatScaled := scaledGroups.flatten
    let qwData := flatScaled.foldl (fun acc bytes => acc.append bytes)
                    (ByteArray.emptyWithCapacity (flatScaled.length * Dtype.float32.itemsize))
    -- pack scale bytes into a ByteArray for scales tensor
    let scaleData := ByteArray.mk (scaleBytes.toArray)
    -- qW has same shape as x, scales has last dim divided by groupSize
    let scalesShape := TensorLib.Shape.mk (x.shape.val.dropLast ++ [lastDim / groupSize])
    return (
      { dtype := .float32, shape := x.shape, data := qwData },
      { dtype := .float8_e8m0, shape := scalesShape, data := scaleData }
    )

section Test

open TensorLib.Tensor.Format.Tree

#guard
  let e := Tensor.ofIntList! Dtype.int32 [10,20,30]
  (e.intAtDimIndex [0], e.intAtDimIndex [1], e.intAtDimIndex [2]) ==
    (.ok 10, .ok 20, .ok 30)

#guard
  (do
    let e <- Tensor.ofIntList Dtype.int32 [10,20,30,40,50,60]
    let e2 <- e.reshape (Shape.mk [2,3])
    let a00 <- e2.intAtDimIndex [0,0]
    let a01 <- e2.intAtDimIndex [0,1]
    let a10 <- e2.intAtDimIndex [1,0]
    let a12 <- e2.intAtDimIndex [1,2]
    return (a00,a01,a10,a12)) ==
    (.ok (10,20,40,60))

#guard
  let arr := arrayScalarNat! Dtype.uint8 5
  let t := arr.toNatTree!
  t == .root [5]

#guard
  let arr := (arange! Dtype.uint16 10).reshape! (Shape.mk [2, 5])
  let t := arr.toNatTree!
  t == node [root [0, 1, 2, 3, 4], root [5, 6, 7, 8, 9]]

#guard (zeros Dtype.float64 $ Shape.mk [2, 2]).nbytes == 2 * 2 * 8
#guard (zeros Dtype.float64 $ Shape.mk [2, 2]).data.toList.count 0 == 2 * 2 * 8

#guard
  let t1 := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t2 := t1.broadcastTo! (Shape.mk [2, 2, 3])
  let tree := t2.toNatTree!
  let n1 := node [ root [0, 1, 2], root [3, 4, 5] ]
  let tree' := node [ n1, n1 ]
  tree == tree'

#guard
  let t1 := (arange! Dtype.uint8 8).reshape! (Shape.mk [2, 1, 1, 4])
  let t2 := t1.broadcastTo! (Shape.mk [2, 3, 3, 4])
  let tree := t2.toNatTree!
  let r1 := root [0, 1, 2, 3]
  let r2 := root [4, 5, 6, 7]
  let n1 := node [ r1, r1, r1 ]
  let n2 := node [ r2, r2, r2 ]
  let tree' := node [ node [ n1, n1, n1 ], node [n2, n2, n2] ]
  tree == tree'

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.uint64
  let t1 := t1.astype! Dtype.uint32
  let t1 := t1.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.int8
  let t1 := t1.astype! Dtype.uint32
  let t1 := t1.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.float32
  let t1 := t1.astype! Dtype.float64
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

-- toNpy rejects e3m4 (V1 ambiguity guard)
#guard match (Tensor.zeros .float8_e3m4 (Shape.mk [2])).toNpy with | .error _ => true | .ok _ => false
-- toNpy accepts e4m3 (not blocked)
#guard match (Tensor.zeros .float8_e4m3 (Shape.mk [2])).toNpy with | .ok _ => true | .error _ => false
#guard match (Tensor.zeros .float8_e2m5 (Shape.mk [2])).toNpy with | .error _ => true | .ok _ => false
#guard match (Tensor.zeros .float8_e8m0 (Shape.mk [2])).toNpy with | .error _ => true | .ok _ => false



end Test

end Tensor
end TensorLib
