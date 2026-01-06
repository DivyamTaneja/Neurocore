package neuroeye.encoder.core

import chisel3._
import chisel3.util._
import neuroeye.encoder.utils.Globals._

class PixelHandler(memDepth: Int, addrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Incoming pixels
    val pixPackedIn = Input(UInt(PACKED_W.W))
    val pixValueIn  = Input(UInt(8.W))
    val pixValidIn  = Input(Bool())

    // Config
    val Tcfg = Input(UInt(8.W))  // dynamic max timesteps

    // Output to PagedMemory
    val memWritePage = Output(UInt(log2Ceil(MAX_T).W))
    val memWriteAddr = Output(UInt(addrWidth.W))
    val memWriteData = Output(UInt(PACKED_W.W))
    val memWriteEn   = Output(Bool())
  })

  // -----------------------------------------------------------
  // Compute timestep for this pixel (private helper)
  // -----------------------------------------------------------
  private def computeT(x: UInt, Tcfg: UInt): UInt = {
    ((255.U - x) * (Tcfg - 1.U) + 127.U) / 255.U
  }

  // -----------------------------------------------------------
  // Compute timestep
  // -----------------------------------------------------------
  val tStep = computeT(io.pixValueIn, io.Tcfg)

  // -----------------------------------------------------------
  // Forward pixel + timestep to PagedMemory
  // -----------------------------------------------------------
  io.memWritePage := tStep
  io.memWriteAddr := io.pixPackedIn(addrWidth-1, 0) // truncated to addrWidth
  io.memWriteData := io.pixPackedIn
  io.memWriteEn   := io.pixValidIn
}