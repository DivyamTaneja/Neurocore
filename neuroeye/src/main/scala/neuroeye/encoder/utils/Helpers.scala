package neuroeye.encoder.utils

import chisel3._
//import chisel3.util._

object Helpers {

  import Globals._

  /** Pack (row, col, ch) → UInt */
  def packPixel(row: UInt, col: UInt, ch: UInt): UInt = {
    (row << ROW_LSB).asUInt |
      (col << COL_LSB).asUInt |
      (ch  << CH_LSB).asUInt
  }

  /** Unpack helpers */
  def unpackRow(addr: UInt): UInt = addr(ROW_MSB, ROW_LSB)
  def unpackCol(addr: UInt): UInt = addr(COL_MSB, COL_LSB)
  def unpackCh (addr: UInt): UInt = addr(CH_MSB, CH_LSB)
}
