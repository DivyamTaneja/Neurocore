package neuroeye.encoder.core.memory.core

import chisel3._
import chisel3.util._

class PagedMemory(
                   numPages: Int = 64,
                   pageSize: Int = 256,
                   addrWidth: Int = log2Ceil(256),
                   pageIdWidth: Int = log2Ceil(64)
                 ) extends Module {

  val io = IO(new Bundle {
    // --- External write interface ---
    val tStep      = Input(UInt(pageIdWidth.W))   // timestep = page index
    val pixelAddr  = Input(UInt(addrWidth.W))
    val pixelValid = Input(Bool())

    // (Optional) for spike handler
    val readPage   = Input(UInt(pageIdWidth.W))
    val readAddr   = Input(UInt(addrWidth.W))
    val readData   = Output(UInt(32.W))
  })

  // -----------------------
  // Page Storage (each page has pageSize entries of UInt(32))
  // -----------------------
  val mem = SyncReadMem(numPages * pageSize, UInt(32.W))

  def pageBase(p: UInt): UInt = p * pageSize.U

  // -----------------------
  // Metadata table for all pages
  // -----------------------
  val metas = RegInit(VecInit(Seq.fill(numPages)(
    PageMeta.default(addrWidth, pageIdWidth)
  )))

  // -----------------------
  // Allocator for overflow pages
  // -----------------------
  val allocator = Module(new PageAllocator(numPages, pageIdWidth))

  allocator.io.allocReq  := false.B
  allocator.io.freeReq   := false.B
  allocator.io.freePage  := 0.U

  // -----------------------
  // WRITE: place pixelAddr in circular buffer of page = tStep
  // -----------------------
  val curPage = io.tStep
  val curMeta = metas(curPage)
  val nextHead = curMeta.headPtr + 1.U

  val overflow = nextHead === curMeta.tailPtr

  // Helper to initialise a page on its first write
  def initPage(p: UInt): Unit = {
    metas(p).valid   := true.B
    metas(p).headPtr := 1.U
    metas(p).tailPtr := 0.U
    mem.write(pageBase(p), io.pixelAddr)
  }

  // Allocate new page on overflow
  when(io.pixelValid) {
    when(!curMeta.valid) {
      // First time writing to this timestep
      initPage(curPage)
    }
    .elsewhen(overflow) {
      // Need new page
      allocator.io.allocReq := true.B
      when(allocator.io.allocDone) {
        val newP = allocator.io.allocResp
        // chain to next page
        metas(curPage).nextPage := newP
        initPage(newP)
      }
    }
      .otherwise {
        // Normal circular write
        val wIndex = pageBase(curPage) + curMeta.headPtr
        mem.write(wIndex, io.pixelAddr)
        metas(curPage).headPtr := nextHead
      }
  }

  // -----------------------
  // READ (SpikeHandler)
  // -----------------------
  val rIndex = pageBase(io.readPage) + io.readAddr
  io.readData := mem.read(rIndex, true.B)
}