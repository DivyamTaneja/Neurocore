package neuroeye.encoder.core.memory.core

import chisel3._

/** Metadata for a single page */
class PageMeta(addrWidth: Int, pageIdWidth: Int) extends Bundle {
  val headPtr   = UInt(addrWidth.W)      // write pointer (circular)
  val tailPtr   = UInt(addrWidth.W)      // read pointer  (circular)
  val valid     = Bool()                 // page is active
  val nextPage  = UInt(pageIdWidth.W)    // linked list: next page index

  override def cloneType =
    new PageMeta(addrWidth, pageIdWidth).asInstanceOf[this.type]
}

/** Default metadata initializer */
object PageMeta {
  def default(addrWidth: Int, pageIdWidth: Int): PageMeta = {
    val m = Wire(new PageMeta(addrWidth, pageIdWidth))
    m.headPtr  := 0.U
    m.tailPtr  := 0.U
    m.valid    := false.B
    m.nextPage := 0.U
    m
  }
}