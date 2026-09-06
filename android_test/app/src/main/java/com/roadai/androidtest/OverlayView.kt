package com.roadai.androidtest

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.roadai.androidtest.detection.Detection
import com.roadai.androidtest.detection.DetectionType
import com.roadai.androidtest.detection.PerceptionResult
import kotlin.math.min

/**
 * Draws the FUSED detections on top of the camera preview.
 *
 * It consumes the unified [PerceptionResult] only — it has no idea which model
 * produced which box, which is exactly the point: one perception system, not
 * two overlays fighting each other.
 *
 * Detections arrive in SOURCE BITMAP pixel coordinates. This view maps them
 * into its own space using FIT_CENTER geometry — the same geometry PreviewView
 * is configured with in [MainActivity]. If that scale type changes to
 * FILL_CENTER, this mapping must change too or every box lands wrong.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var sourceWidth = 0
    private var sourceHeight = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = 38f
        isFakeBoldText = true
    }

    /**
     * Colour comes from the NORMALIZED type, never from a class index — two
     * models number their classes independently, so indices would collide.
     */
    private fun colorFor(type: DetectionType): Int = when (type) {
        DetectionType.POTHOLE -> Color.parseColor("#FF1744") // red
        DetectionType.CRACK -> Color.parseColor("#00E5FF")   // cyan
        DetectionType.OTHER -> Color.parseColor("#BDBDBD")   // grey
    }

    fun setResult(result: PerceptionResult) {
        detections = result.detections
        sourceWidth = result.sourceWidth
        sourceHeight = result.sourceHeight
        postInvalidateOnAnimation()
    }

    fun clear() {
        detections = emptyList()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() || sourceWidth <= 0 || sourceHeight <= 0) return

        // FIT_CENTER: uniform scale, centred, letterboxed inside this view.
        val scale = min(width.toFloat() / sourceWidth, height.toFloat() / sourceHeight)
        val offsetX = (width - sourceWidth * scale) / 2f
        val offsetY = (height - sourceHeight * scale) / 2f

        for (d in detections) {
            val rect = RectF(
                d.x1 * scale + offsetX,
                d.y1 * scale + offsetY,
                d.x2 * scale + offsetX,
                d.y2 * scale + offsetY
            )

            val color = colorFor(d.type)
            boxPaint.color = color
            canvas.drawRect(rect, boxPaint)

            // Model filenames are never surfaced — the user sees one system.
            val label = "${d.displayLabel} ${(d.confidence * 100).toInt()}%"
            val textWidth = labelTextPaint.measureText(label)
            val textHeight = labelTextPaint.textSize + 12f

            // Keep the label on screen when the box touches the top edge.
            val labelTop = if (rect.top - textHeight < 0) rect.top else rect.top - textHeight
            labelBgPaint.color = color
            canvas.drawRect(
                rect.left,
                labelTop,
                rect.left + textWidth + 16f,
                labelTop + textHeight,
                labelBgPaint
            )
            canvas.drawText(label, rect.left + 8f, labelTop + textHeight - 10f, labelTextPaint)
        }
    }
}
