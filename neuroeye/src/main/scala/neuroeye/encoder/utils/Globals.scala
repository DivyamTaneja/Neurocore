package neuroeye.encoder.utils

//import chisel3._
import chisel3.util._

object Globals {

  // -------------------------------
  // Configuration Limits
  // -------------------------------
  val MAX_IMG_W = 128
  val MAX_IMG_H = 128
  val MAX_CH    = 4
  val MAX_T     = 32

  // -------------------------------
  // Bit Widths
  // -------------------------------
  val ROW_W = log2Ceil(MAX_IMG_H)
  val COL_W = log2Ceil(MAX_IMG_W)
  val CH_W  = log2Ceil(MAX_CH)

  val PACKED_W = ROW_W + COL_W + CH_W

  // -------------------------------
  // Bit Field Positions
  // -------------------------------
  val CH_LSB  = 0
  val CH_MSB  = CH_LSB + CH_W - 1

  val COL_LSB = CH_MSB + 1
  val COL_MSB = COL_LSB + COL_W - 1

  val ROW_LSB = COL_MSB + 1
  val ROW_MSB = ROW_LSB + ROW_W - 1
}
