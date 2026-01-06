package neuroeye.encoder

import chisel3._
import chisel3.util._
import neuroeye.encoder.utils._
import neuroeye.encoder.core._

/**
 * Encoder subsystem IO.
 * This is the public interface that the SoC connects to.
 */
class Interface extends Bundle {
  val pixPackedIn = Input(UInt(Globals.PACKED_W.W))
  val pixValueIn  = Input(UInt(8.W))
  val pixValidIn  = Input(Bool())

  val Tcfg = Input(UInt(8.W))

  val reqRowStart = Input(UInt(Globals.ROW_W.W))
  val reqRowEnd   = Input(UInt(Globals.ROW_W.W))
  val reqT        = Input(UInt(log2Ceil(Globals.MAX_T).W))
  val reqValid    = Input(Bool())

  val respPackedAddr = Output(UInt(Globals.PACKED_W.W))
  val respValid      = Output(Bool())

  val frameDone = Output(Bool())

  override def cloneType = (new Interface).asInstanceOf[this.type]
}

/**
 * Top-level Encoder (Paged TTFS Encoder)
 */
class Encoder extends Module {
  val io = IO(new Interface)

  // ------------------------------------------------------------
  // Submodules
  // ------------------------------------------------------------
  val pageMem   = Module(new PageMemory)
  val meta      = Module(new PageMetadata)
  val allocator = Module(new PageAllocator)
  val trav      = Module(new TraversalFsm)

  // ------------------------------------------------------------
  // Wiring: Write path
  // ------------------------------------------------------------
  pageMem.io.pixValueIn  := io.pixValueIn
  pageMem.io.pixPackedIn := io.pixPackedIn
  pageMem.io.pixValidIn  := io.pixValidIn

  pageMem.io.Tcfg := io.Tcfg

  // Allocator wiring
  allocator.io.allocReq := pageMem.io.pageAllocReq
  pageMem.io.allocPage  := allocator.io.allocPage

  allocator.io.freeReq := meta.io.freePageReq

  // Metadata wiring
  meta.io.writeReq := pageMem.io.writeMetaReq
  pageMem.io.meta   := meta.io.writeMetaResp

  // ------------------------------------------------------------
  // Wiring: Read path (traversal)
  // ------------------------------------------------------------
  trav.io.reqRowStart := io.reqRowStart
  trav.io.reqRowEnd   := io.reqRowEnd
  trav.io.reqT        := io.reqT
  trav.io.reqValid    := io.reqValid

  // traversal reads from PageMemory
  pageMem.io.readReq    := trav.io.readReq
  trav.io.readData      := pageMem.io.readData

  // returning outputs
  io.respPackedAddr := trav.io.respPackedAddr
  io.respValid      := trav.io.respValid

  // ------------------------------------------------------------
  // Frame-done indicator
  // ------------------------------------------------------------
  io.frameDone := pageMem.io.frameDone
}
