from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..config import ModelConfig

BACKGROUND = QColor("#10131a")
PANEL = QColor("#171c26")
GRID = QColor("#2a3342")
TEXT = QColor("#dce5f2")
MUTED = QColor("#8492a6")
CYAN = QColor("#4fd1c5")
PURPLE = QColor("#a78bfa")
ORANGE = QColor("#f6ad55")


class MetricsChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.train_points: list[tuple[int, float]] = []
        self.validation_points: list[tuple[int, float]] = []

    def clear(self) -> None:
        self.train_points.clear()
        self.validation_points.clear()
        self.update()

    def add_point(self, step: int, train_loss: float, validation_loss: float | None) -> None:
        self.train_points.append((step, train_loss))
        if validation_loss is not None:
            self.validation_points.append((step, validation_loss))
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), PANEL)
        plot = QRectF(52, 28, max(10, self.width() - 72), max(10, self.height() - 66))
        painter.setPen(QPen(GRID, 1))
        for index in range(5):
            y = plot.top() + plot.height() * index / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        all_points = self.train_points + self.validation_points
        painter.setPen(TEXT)
        painter.drawText(14, 19, "Loss")
        if not all_points:
            painter.setPen(MUTED)
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "学習を始めると曲線が表示されます")
            return

        max_step = max(point[0] for point in all_points)
        values = [point[1] for point in all_points if math.isfinite(point[1])]
        min_value = max(0.0, min(values) * 0.9)
        max_value = max(values) * 1.1
        if max_value <= min_value:
            max_value = min_value + 1.0

        painter.setFont(QFont("Sans Serif", 8))
        painter.setPen(MUTED)
        for index in range(5):
            value = max_value - (max_value - min_value) * index / 4
            y = plot.top() + plot.height() * index / 4
            painter.drawText(QRectF(2, y - 8, 44, 16), Qt.AlignmentFlag.AlignRight, f"{value:.2f}")
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 6, plot.width(), 18),
            Qt.AlignmentFlag.AlignRight,
            f"step {max_step}",
        )

        self._draw_series(painter, plot, self.train_points, CYAN, max_step, min_value, max_value)
        self._draw_series(
            painter, plot, self.validation_points, PURPLE, max_step, min_value, max_value
        )
        painter.setFont(QFont("Sans Serif", 9))
        painter.setPen(CYAN)
        painter.drawText(plot.left() + 8, plot.top() + 17, "● Train")
        painter.setPen(PURPLE)
        painter.drawText(plot.left() + 82, plot.top() + 17, "● Validation")

    @staticmethod
    def _draw_series(
        painter: QPainter,
        plot: QRectF,
        points: list[tuple[int, float]],
        color: QColor,
        max_step: int,
        min_value: float,
        max_value: float,
    ) -> None:
        if not points:
            return
        path = QPainterPath()
        for index, (step, value) in enumerate(points):
            x = plot.left() + plot.width() * step / max(1, max_step)
            y = plot.bottom() - plot.height() * (value - min_value) / (max_value - min_value)
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(color, 2.2))
        painter.drawPath(path)


class AttentionHeatmap(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.matrix: list[list[float]] = []
        self.tokens: list[str] = []
        self.title = "Attention（最終層・Head平均）"

    def set_attention(self, matrix: list[list[float]], tokens: list[str]) -> None:
        self.matrix = matrix
        self.tokens = tokens
        self.update()

    def clear(self) -> None:
        self.matrix = []
        self.tokens = []
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), PANEL)
        painter.setPen(TEXT)
        painter.drawText(12, 20, self.title)
        if not self.matrix:
            painter.setPen(MUTED)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Attention待機中")
            return
        count = min(len(self.matrix), len(self.tokens))
        left = 48.0
        top = 52.0
        size = min(self.width() - left - 12, self.height() - top - 34)
        cell = size / max(1, count)
        font = QFont("Sans Serif", max(6, min(9, int(cell * 0.55))))
        painter.setFont(font)
        for row in range(count):
            for column in range(count):
                value = max(0.0, min(1.0, float(self.matrix[row][column])))
                color = QColor.fromRgbF(
                    0.12 + value * 0.45,
                    0.16 + value * 0.32,
                    0.24 + value * 0.60,
                )
                painter.fillRect(QRectF(left + column * cell, top + row * cell, cell, cell), color)
        painter.setPen(MUTED)
        label_stride = max(1, math.ceil(count / 12))
        for index, token in enumerate(self.tokens[:count]):
            if index % label_stride:
                continue
            label = token[:5]
            painter.drawText(
                QRectF(left - 43, top + index * cell, 39, cell),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.save()
            painter.translate(left + index * cell + cell * 0.65, top - 5)
            painter.rotate(-55)
            painter.drawText(0, 0, label)
            painter.restore()
        painter.drawText(
            QRectF(left, self.height() - 24, size, 16),
            Qt.AlignmentFlag.AlignCenter,
            "参照されるToken →",
        )


class ArchitectureView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config: ModelConfig | None = None
        self.setMinimumHeight(260)

    def set_config(self, config: ModelConfig | None) -> None:
        self.config = config
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), PANEL)
        config = self.config
        if config is None:
            painter.setPen(MUTED)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "モデル未構築")
            return
        blocks = [
            ("入力Token ID", f"長さ ≤ {config.context_length}", QColor("#334155")),
            ("Token + Position Embedding", f"{config.embedding_dim} 次元", QColor("#164e63")),
            (
                f"Transformer Block × {config.num_layers}",
                f"Causal Attention {config.num_heads} heads + FFN",
                QColor("#4c1d95"),
            ),
            ("LayerNorm + Linear", f"→ {config.vocab_size} 語彙", QColor("#7c2d12")),
            ("Softmax", "次Tokenの確率分布", QColor("#14532d")),
        ]
        margin = 18.0
        gap = 12.0
        block_height = (self.height() - 2 * margin - gap * (len(blocks) - 1)) / len(blocks)
        for index, (title, subtitle, color) in enumerate(blocks):
            y = margin + index * (block_height + gap)
            rect = QRectF(margin, y, self.width() - 2 * margin, block_height)
            gradient = QLinearGradient(rect.topLeft(), rect.topRight())
            gradient.setColorAt(0, color)
            gradient.setColorAt(1, color.lighter(125))
            painter.setBrush(gradient)
            painter.setPen(QPen(color.lighter(155), 1))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(TEXT)
            painter.setFont(QFont("Sans Serif", 10, QFont.Weight.DemiBold))
            painter.drawText(QRectF(rect.left() + 12, y + 5, rect.width() - 24, 18), title)
            painter.setPen(QColor("#c2cad6"))
            painter.setFont(QFont("Sans Serif", 8))
            painter.drawText(QRectF(rect.left() + 12, y + 23, rect.width() - 24, 16), subtitle)
            if index < len(blocks) - 1:
                painter.setPen(QPen(ORANGE, 2))
                center = rect.center().x()
                painter.drawLine(
                    QPointF(center, rect.bottom()), QPointF(center, rect.bottom() + gap)
                )
