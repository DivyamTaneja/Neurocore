package neuroeye.encoder.core.memory.core

import chisel3._
import chisel3.util._

/** Simple free-list based allocator */
class PageAllocator(numPages: Int, pageIdWidth: Int) extends Module {
  val io = IO(new Bundle {
    val allocReq  = Input(Bool())
    val allocResp = Output(UInt(pageIdWidth.W))
    val allocDone = Output(Bool())

    val freeReq   = Input(Bool())
    val freePage  = Input(UInt(pageIdWidth.W))
  })

  // Free list implemented as a queue of page IDs
  val freeList = RegInit(VecInit((0 until numPages).map(_.U(pageIdWidth.W))))
  val freeHead = RegInit(0.U(log2Ceil(numPages).W))
  val freeTail = RegInit(numPages.U(log2Ceil(numPages+1).W)) // initially full

  def freeCount = freeTail - freeHead

  // Allocate
  val allocValid = io.allocReq && (freeCount =/= 0.U)
  io.allocDone := allocValid
  io.allocResp := freeList(freeHead)

  when (allocValid) {
    freeHead := freeHead + 1.U
  }

  // Free a page
  when(io.freeReq) {
    freeList(freeTail(log2Ceil(numPages)-1, 0)) := io.freePage
    freeTail := freeTail + 1.U
  }
}