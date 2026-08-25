from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..accelerators import (
    accelerator_report,
    describe_device,
    detect_accelerators,
    select_device,
)
from ..config import ModelConfig, TrainingConfig
from ..model import TinyGPT
from ..tokenizer import BPETokenizer
from ..training import LanguageModelTrainer, load_checkpoint
from .visualizations import ArchitectureView, AttentionHeatmap, MetricsChart
from .workers import GenerationWorker, TrainingWorker

APP_STYLE = """
QWidget { background: #10131a; color: #dce5f2; font-size: 13px; }
QMainWindow, QTabWidget::pane { background: #10131a; }
QTabWidget::pane { border: 1px solid #2a3342; }
QTabBar::tab { background: #171c26; padding: 10px 18px; margin-right: 2px; color: #8492a6; }
QTabBar::tab:selected { color: #f3f6fa; border-bottom: 2px solid #4fd1c5; }
QGroupBox { border: 1px solid #2a3342; border-radius: 8px; margin-top: 12px; padding: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #aebbd0; }
QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
  background: #171c26; border: 1px solid #334155; border-radius: 5px; padding: 5px; selection-background-color: #285e61;
}
QPushButton { background: #253044; border: 1px solid #3a4961; border-radius: 6px; padding: 8px 14px; }
QPushButton:hover { background: #30415c; }
QPushButton:disabled { color: #596579; background: #171c26; }
QPushButton#primary { background: #16736d; border-color: #319795; font-weight: 600; }
QPushButton#danger { background: #672c36; border-color: #9b4150; }
QProgressBar { border: 1px solid #334155; border-radius: 5px; text-align: center; background: #171c26; }
QProgressBar::chunk { background: #319795; border-radius: 4px; }
QHeaderView::section { background: #202838; color: #aebbd0; padding: 6px; border: none; }
QScrollBar:vertical { background: #10131a; width: 10px; }
QScrollBar::handle:vertical { background: #334155; border-radius: 5px; min-height: 24px; }
"""


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—") -> None:
        super().__init__()
        self.setStyleSheet(
            "QFrame { background: #171c26; border: 1px solid #2a3342; border-radius: 8px; }"
        )
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #8492a6; border: none;")
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Sans Serif", 16, QFont.Weight.DemiBold))
        self.value_label.setStyleSheet("color: #f3f6fa; border: none;")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Deep Learning Studio — GPTをゼロから学ぶ")
        self.resize(1380, 900)
        self.setMinimumSize(1040, 720)
        self.setStyleSheet(APP_STYLE)

        self.tokenizer: BPETokenizer | None = None
        self.model: TinyGPT | None = None
        self.trainer: LanguageModelTrainer | None = None
        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self.generation_thread: QThread | None = None
        self.generation_worker: GenerationWorker | None = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.data_tab = self._build_data_tab()
        self.training_tab = self._build_training_tab()
        self.generation_tab = self._build_generation_tab()
        self.tabs.addTab(self.data_tab, "1  データと設定")
        self.tabs.addTab(self.training_tab, "2  学習の可視化")
        self.tabs.addTab(self.generation_tab, "3  文章生成")
        self._create_menu()
        self._apply_preset(0)
        self._refresh_device_label()
        self.statusBar().showMessage("文章を入力して、最初のモデルを作りましょう")

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")
        open_action = QAction("チェックポイントを開く…", self)
        open_action.triggered.connect(self.open_checkpoint)
        self.save_action = QAction("チェックポイントを保存…", self)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_checkpoint)
        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _build_data_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel("自分の文章から、モデルの知識を作る")
        title.setFont(QFont("Sans Serif", 19, QFont.Weight.Bold))
        description = QLabel(
            "ここへ入力したデータだけでBPE語彙とGPT型モデルを学習します。データは外部へ送信されません。"
        )
        description.setStyleSheet("color: #8492a6;")
        layout.addWidget(title)
        layout.addWidget(description)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        corpus_group = QGroupBox("学習コーパス")
        corpus_layout = QVBoxLayout(corpus_group)
        self.corpus_editor = QPlainTextEdit()
        self.corpus_editor.setPlaceholderText(
            "例：自分で書いた会話、物語、説明文などを貼り付けます。\n"
            "同じ言い回しやパターンが何度も現れるデータほど、小さなモデルでも学習を観察しやすくなります。"
        )
        self.corpus_editor.textChanged.connect(self._update_corpus_stats)
        self.corpus_stats = QLabel("0文字")
        self.corpus_stats.setStyleSheet("color: #8492a6;")
        corpus_layout.addWidget(self.corpus_editor)
        corpus_layout.addWidget(self.corpus_stats)
        left_layout.addWidget(corpus_group)

        tokenizer_group = QGroupBox("BPEトークナイザー")
        tokenizer_layout = QFormLayout(tokenizer_group)
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(32, 4096)
        self.vocab_size_spin.setValue(512)
        self.min_frequency_spin = QSpinBox()
        self.min_frequency_spin.setRange(1, 100)
        self.min_frequency_spin.setValue(2)
        self.build_tokenizer_button = QPushButton("トークナイザーを構築")
        self.build_tokenizer_button.setObjectName("primary")
        self.build_tokenizer_button.clicked.connect(self.build_tokenizer)
        tokenizer_layout.addRow("目標語彙数", self.vocab_size_spin)
        tokenizer_layout.addRow("最小ペア出現数", self.min_frequency_spin)
        tokenizer_layout.addRow(self.build_tokenizer_button)
        left_layout.addWidget(tokenizer_group)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        model_group = QGroupBox("GPTモデル設定")
        model_layout = QFormLayout(model_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["学習用ミニ（CPU向け）", "標準", "本格（GPU推奨）"])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.context_spin = QSpinBox()
        self.context_spin.setRange(16, 1024)
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItems(["64", "96", "128", "192", "256", "384"])
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 16)
        self.heads_spin = QSpinBox()
        self.heads_spin.setRange(1, 16)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.05)
        model_layout.addRow("プリセット", self.preset_combo)
        model_layout.addRow("Context length", self.context_spin)
        model_layout.addRow("Embedding次元", self.embedding_combo)
        model_layout.addRow("Transformer層", self.layers_spin)
        model_layout.addRow("Attention Heads", self.heads_spin)
        model_layout.addRow("Dropout", self.dropout_spin)
        right_layout.addWidget(model_group)

        training_group = QGroupBox("学習設定")
        training_layout = QFormLayout(training_group)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 1_000_000)
        self.steps_spin.setValue(1000)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 0.1)
        self.learning_rate_spin.setValue(0.0003)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.eval_interval_spin = QSpinBox()
        self.eval_interval_spin.setRange(1, 10000)
        self.eval_interval_spin.setValue(25)
        self.device_combo = QComboBox()
        self._populate_accelerators()
        self.device_combo.currentIndexChanged.connect(self._refresh_device_label)
        self.device_label = QLabel()
        self.device_label.setStyleSheet("color: #4fd1c5;")
        self.diagnostics_button = QPushButton("GPU・アクセラレータ診断を表示")
        self.diagnostics_button.clicked.connect(self.show_accelerator_diagnostics)
        training_layout.addRow("学習Step", self.steps_spin)
        training_layout.addRow("Batch size", self.batch_spin)
        training_layout.addRow("Learning rate", self.learning_rate_spin)
        training_layout.addRow("可視化間隔", self.eval_interval_spin)
        training_layout.addRow("演算デバイス", self.device_combo)
        training_layout.addRow("検出結果", self.device_label)
        training_layout.addRow(self.diagnostics_button)
        right_layout.addWidget(training_group)

        self.start_button = QPushButton("モデルを作成して学習を開始")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        right_layout.addWidget(self.start_button)
        right_layout.addStretch()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([760, 480])
        layout.addWidget(splitter)
        return root

    def _build_training_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        header = QHBoxLayout()
        title = QLabel("モデル内部で起きていること")
        title.setFont(QFont("Sans Serif", 19, QFont.Weight.Bold))
        self.training_status = QLabel("待機中")
        self.training_status.setStyleSheet("color: #4fd1c5;")
        self.stop_button = QPushButton("学習を停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.training_status)
        header.addWidget(self.stop_button)
        layout.addLayout(header)

        cards = QHBoxLayout()
        self.step_card = MetricCard("Step")
        self.loss_card = MetricCard("Train Loss")
        self.validation_card = MetricCard("Validation Loss")
        self.perplexity_card = MetricCard("Perplexity")
        self.speed_card = MetricCard("Tokens / sec")
        for card in (
            self.step_card,
            self.loss_card,
            self.validation_card,
            self.perplexity_card,
            self.speed_card,
        ):
            cards.addWidget(card)
        layout.addLayout(cards)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.metrics_chart = MetricsChart()
        left_layout.addWidget(self.metrics_chart, 2)
        self.attention_view = AttentionHeatmap()
        left_layout.addWidget(self.attention_view, 3)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.architecture_view = ArchitectureView()
        right_layout.addWidget(self.architecture_view, 3)
        detail_group = QGroupBox("この瞬間の処理")
        detail_layout = QVBoxLayout(detail_group)
        self.process_label = QLabel(
            "学習では、入力Tokenから次のTokenを予測し、正解との誤差を逆伝播して全パラメータを更新します。"
        )
        self.process_label.setWordWrap(True)
        self.process_label.setStyleSheet("color: #aebbd0;")
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(300)
        detail_layout.addWidget(self.process_label)
        detail_layout.addWidget(self.training_log)
        right_layout.addWidget(detail_group, 2)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([820, 430])
        layout.addWidget(splitter)
        return root

    def _build_generation_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel("学習したモデルに文章を書かせる")
        title.setFont(QFont("Sans Serif", 19, QFont.Weight.Bold))
        layout.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        prompt_group = QGroupBox("プロンプト")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_editor = QPlainTextEdit()
        self.prompt_editor.setPlaceholderText("書き出しを入力してください")
        prompt_layout.addWidget(self.prompt_editor)
        left_layout.addWidget(prompt_group)
        controls = QGroupBox("Sampling設定")
        controls_layout = QFormLayout(controls)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 1000)
        self.max_tokens_spin.setValue(120)
        self.auto_stop_checkbox = QCheckBox("文章の終わりで自動停止")
        self.auto_stop_checkbox.setChecked(True)
        self.auto_stop_checkbox.setToolTip(
            "EOS、句点、感嘆符、疑問符、改行を予測したら停止します。最大Token数は安全上の上限です。"
        )
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.05, 2.0)
        self.temperature_spin.setValue(0.8)
        self.temperature_spin.setSingleStep(0.05)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 1000)
        self.top_k_spin.setValue(40)
        self.repetition_spin = QDoubleSpinBox()
        self.repetition_spin.setRange(1.0, 2.0)
        self.repetition_spin.setValue(1.05)
        self.repetition_spin.setSingleStep(0.05)
        controls_layout.addRow("最大Token数", self.max_tokens_spin)
        controls_layout.addRow(self.auto_stop_checkbox)
        controls_layout.addRow("Temperature", self.temperature_spin)
        controls_layout.addRow("Top-K", self.top_k_spin)
        controls_layout.addRow("反復Penalty", self.repetition_spin)
        left_layout.addWidget(controls)
        self.generate_button = QPushButton("生成を開始")
        self.generate_button.setObjectName("primary")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.start_generation)
        left_layout.addWidget(self.generate_button)
        output_group = QGroupBox("生成結果（Tokenごとに更新）")
        output_layout = QVBoxLayout(output_group)
        self.output_editor = QPlainTextEdit()
        self.output_editor.setReadOnly(True)
        output_layout.addWidget(self.output_editor)
        left_layout.addWidget(output_group, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.generation_attention = AttentionHeatmap()
        self.generation_attention.title = "生成時Attention（最終層・Head平均）"
        right_layout.addWidget(self.generation_attention, 3)
        candidates_group = QGroupBox("次Token候補")
        candidates_layout = QVBoxLayout(candidates_group)
        self.candidate_table = QTableWidget(0, 2)
        self.candidate_table.setHorizontalHeaderLabels(["Token", "確率"])
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.verticalHeader().setVisible(False)
        candidates_layout.addWidget(self.candidate_table)
        right_layout.addWidget(candidates_group, 2)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 520])
        layout.addWidget(splitter)
        return root

    def _apply_preset(self, index: int) -> None:
        presets = [
            (64, "64", 2, 4, 0.10, 500, 16),
            (128, "128", 4, 4, 0.10, 1000, 16),
            (128, "192", 6, 6, 0.10, 2000, 12),
        ]
        context, embedding, layers, heads, dropout, steps, batch = presets[index]
        self.context_spin.setValue(context)
        self.embedding_combo.setCurrentText(embedding)
        self.layers_spin.setValue(layers)
        self.heads_spin.setValue(heads)
        self.dropout_spin.setValue(dropout)
        self.steps_spin.setValue(steps)
        self.batch_spin.setValue(batch)

    def _refresh_device_label(self, *_args) -> None:
        key = self.device_combo.currentData() or "auto"
        try:
            self.device_label.setStyleSheet("color: #4fd1c5;")
            self.device_label.setText(describe_device(select_device(key)))
        except Exception as error:
            self.device_label.setStyleSheet("color: #f6ad55;")
            self.device_label.setText(str(error))

    def _populate_accelerators(self) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("自動選択（推奨）", "auto")
        for item in detect_accelerators():
            marker = "✓" if item.available else "—"
            self.device_combo.addItem(f"{marker} {item.label}", item.key)

    def show_accelerator_diagnostics(self) -> None:
        report = accelerator_report()
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Information)
        message.setWindowTitle("GPU・アクセラレータ診断")
        message.setText(
            "PyTorchと演算デバイスの検出結果です。CUDAが利用不可の場合は「詳細を表示」を確認してください。"
        )
        message.setDetailedText(report)
        message.exec()

    def _update_corpus_stats(self) -> None:
        text = self.corpus_editor.toPlainText()
        lines = text.count("\n") + (1 if text else 0)
        self.corpus_stats.setText(f"{len(text):,}文字 / {lines:,}行")

    def build_tokenizer(self) -> None:
        corpus = self.corpus_editor.toPlainText()
        if len(corpus.strip()) < 20:
            QMessageBox.warning(
                self, "データ不足", "まず20文字以上の学習データを入力してください。"
            )
            return
        try:
            tokenizer = BPETokenizer()
            history = tokenizer.train(
                corpus,
                target_vocab_size=self.vocab_size_spin.value(),
                min_pair_frequency=self.min_frequency_spin.value(),
            )
            self.tokenizer = tokenizer
            encoded = tokenizer.encode(corpus, add_bos=True, add_eos=True)
            ratio = len(corpus) / max(1, len(encoded))
            self.corpus_stats.setText(
                f"{len(corpus):,}文字 → {len(encoded):,} tokens / "
                f"語彙 {tokenizer.vocab_size:,} / 1 token平均 {ratio:.2f}文字"
            )
            self.training_log.clear()
            self.training_log.appendPlainText("BPE語彙構築の過程（頻出ペアを結合）")
            for item in history[-80:]:
                left = item["left"].replace("\n", "\\n")
                right = item["right"].replace("\n", "\\n")
                merged = item["merged"].replace("\n", "\\n")
                self.training_log.appendPlainText(
                    f"merge {item['step']:>3}: {left!r} + {right!r} → {merged!r} "
                    f"({item['frequency']}回)"
                )
            self.training_log.appendPlainText(
                f"完了: 基本文字 {len(tokenizer.base_characters)} / merge {len(tokenizer.merges)}"
            )
            self.start_button.setEnabled(True)
            self.statusBar().showMessage("トークナイザーを構築しました。次はモデルを学習できます。")
            self.tabs.setCurrentWidget(self.training_tab)
        except Exception as error:
            self._show_error("トークナイザー構築に失敗しました", error)

    def _model_config(self) -> ModelConfig:
        assert self.tokenizer is not None
        config = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            context_length=self.context_spin.value(),
            embedding_dim=int(self.embedding_combo.currentText()),
            num_heads=self.heads_spin.value(),
            num_layers=self.layers_spin.value(),
            dropout=self.dropout_spin.value(),
        )
        config.validate()
        return config

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            max_steps=self.steps_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            warmup_steps=min(50, max(1, self.steps_spin.value() // 10)),
            eval_interval=self.eval_interval_spin.value(),
            eval_batches=5,
            device=self.device_combo.currentData() or "auto",
        )

    def start_training(self) -> None:
        if self.tokenizer is None:
            return
        try:
            model = TinyGPT(self._model_config())
            trainer = LanguageModelTrainer(
                model,
                self.tokenizer,
                self.corpus_editor.toPlainText(),
                self._training_config(),
            )
            self.model = model
            self.trainer = trainer
            self.metrics_chart.clear()
            self.attention_view.clear()
            self.architecture_view.set_config(model.config)
            self.training_log.clear()
            self.training_log.appendPlainText(
                f"Model: {model.parameter_count():,} trainable parameters"
            )
            self.training_log.appendPlainText(f"Device: {describe_device(trainer.device)}")
            self.training_log.appendPlainText(
                f"Train tokens: {len(trainer.train_data):,} / Validation tokens: "
                f"{len(trainer.validation_data):,}"
            )
            self._set_training_running(True)
            self.tabs.setCurrentWidget(self.training_tab)

            thread = QThread(self)
            worker = TrainingWorker(trainer)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.metrics.connect(self._on_metrics)
            worker.status.connect(self._on_training_status)
            worker.error.connect(self._on_worker_error)
            worker.finished.connect(self._on_training_finished)
            worker.finished.connect(worker.deleteLater)
            worker.finished.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            self.training_thread = thread
            self.training_worker = worker
            thread.start()
        except Exception as error:
            self._show_error("学習を開始できませんでした", error)

    def _set_training_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self.tokenizer is not None)
        self.build_tokenizer_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.generate_button.setEnabled(not running and self.model is not None)
        self.save_action.setEnabled(not running and self.trainer is not None)
        if running:
            self.progress_bar.setValue(0)

    def stop_training(self) -> None:
        if self.training_worker:
            self.training_status.setText("停止処理中…")
            self.training_worker.stop()
            self.stop_button.setEnabled(False)

    def _on_metrics(self, metrics: dict) -> None:
        step = metrics["step"]
        max_steps = metrics["max_steps"]
        validation = metrics["validation_loss"]
        perplexity = metrics["perplexity"]
        self.step_card.set_value(f"{step:,} / {max_steps:,}")
        self.loss_card.set_value(f"{metrics['train_loss']:.4f}")
        self.validation_card.set_value(f"{validation:.4f}" if validation is not None else "—")
        self.perplexity_card.set_value(f"{perplexity:.2f}" if perplexity is not None else "—")
        self.speed_card.set_value(f"{metrics['tokens_per_second']:,.0f}")
        self.progress_bar.setValue(round(step / max_steps * 1000))
        self.metrics_chart.add_point(step, metrics["train_loss"], validation)
        if metrics["attention"]:
            self.attention_view.set_attention(metrics["attention"], metrics["attention_tokens"])
        self.process_label.setText(
            f"Step {step}: 予測誤差から勾配を計算し、AdamWで重みを更新しました。"
            f"勾配ノルム={metrics['gradient_norm']:.3f}、学習率={metrics['learning_rate']:.2e}。"
            f"可視化サンプルの次Token予測は {metrics['sample_prediction']!r} です。"
        )
        self.training_log.appendPlainText(
            f"step {step:>6} | train {metrics['train_loss']:.4f} | "
            f"val {validation:.4f} | ppl {perplexity:.2f} | "
            f"{metrics['tokens_per_second']:,.0f} tok/s"
        )

    def _on_training_status(self, status: str) -> None:
        self.training_status.setText(status)
        self.statusBar().showMessage(status)

    def _on_training_finished(self) -> None:
        self._set_training_running(False)
        self.training_status.setText("学習完了 / 生成可能")
        self.generate_button.setEnabled(self.model is not None)
        self.save_action.setEnabled(self.trainer is not None)
        self.training_worker = None
        self.training_thread = None

    def _on_worker_error(self, details: str) -> None:
        self.training_log.appendPlainText(details)
        QMessageBox.critical(self, "処理中にエラーが発生しました", details.splitlines()[-1])

    def start_generation(self) -> None:
        if self.model is None or self.tokenizer is None or self.generation_thread is not None:
            return
        prompt = self.prompt_editor.toPlainText()
        self.output_editor.setPlainText(prompt)
        self.candidate_table.setRowCount(0)
        self.generation_attention.clear()
        self.generate_button.setEnabled(False)
        thread = QThread(self)
        worker = GenerationWorker(
            self.model,
            self.tokenizer,
            prompt,
            self.max_tokens_spin.value(),
            self.temperature_spin.value(),
            min(self.top_k_spin.value(), self.tokenizer.vocab_size),
            self.repetition_spin.value(),
            self.auto_stop_checkbox.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.token.connect(self._on_generated_token)
        worker.finished.connect(self._on_generation_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_generation_error)
        worker.error.connect(worker.deleteLater)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self.generation_thread = thread
        self.generation_worker = worker
        thread.start()

    def _on_generated_token(self, update: dict) -> None:
        cursor = self.output_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(update["text"])
        self.output_editor.setTextCursor(cursor)
        candidates = update["candidates"]
        self.candidate_table.setRowCount(len(candidates))
        for row, (token, probability) in enumerate(candidates):
            self.candidate_table.setItem(row, 0, QTableWidgetItem(token))
            self.candidate_table.setItem(row, 1, QTableWidgetItem(f"{probability * 100:.2f}%"))
        if update["attention"]:
            self.generation_attention.set_attention(update["attention"], update["attention_tokens"])

    def _on_generation_finished(self, text: str) -> None:
        self.output_editor.setPlainText(text)
        self.generate_button.setEnabled(True)
        self.generation_worker = None
        self.generation_thread = None
        self.statusBar().showMessage("文章生成が完了しました")

    def _on_generation_error(self, details: str) -> None:
        self.generate_button.setEnabled(True)
        self.generation_worker = None
        self.generation_thread = None
        QMessageBox.critical(self, "生成に失敗しました", details.splitlines()[-1])

    def save_checkpoint(self) -> None:
        if self.trainer is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "チェックポイントを保存", "model.pt", "PyTorch checkpoint (*.pt)"
        )
        if not path:
            return
        try:
            self.trainer.save_checkpoint(path)
            self.statusBar().showMessage(f"保存しました: {path}")
        except Exception as error:
            self._show_error("保存に失敗しました", error)

    def open_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "チェックポイントを開く", "", "PyTorch checkpoint (*.pt *.pth)"
        )
        if not path:
            return
        try:
            model, tokenizer, checkpoint = load_checkpoint(path, device="auto")
            self.model = model
            self.tokenizer = tokenizer
            self.trainer = None
            self.architecture_view.set_config(model.config)
            self.generate_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.tabs.setCurrentWidget(self.generation_tab)
            self.statusBar().showMessage(
                f"{Path(path).name} を読み込みました（step {checkpoint.get('step', 0)}）"
            )
        except Exception as error:
            self._show_error("読み込みに失敗しました", error)

    def _show_error(self, title: str, error: Exception) -> None:
        details = "".join(traceback.format_exception(error))
        message = QMessageBox(QMessageBox.Icon.Critical, title, str(error), parent=self)
        message.setDetailedText(details)
        message.exec()

    def closeEvent(self, event) -> None:
        if self.training_worker is not None:
            answer = QMessageBox.question(
                self,
                "学習中です",
                "学習を停止して終了しますか？未保存の重みは失われます。",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.training_worker.stop()
            if self.training_thread is not None and not self.training_thread.wait(30_000):
                QMessageBox.warning(
                    self,
                    "停止処理中",
                    "GPU処理がまだ終了していません。停止完了後にもう一度終了してください。",
                )
                event.ignore()
                return
        event.accept()
