package com.example.medicalimageapp;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    // ========== UI组件 ==========
    private ImageView imageView;
    private Button btnSelect;
    private Button btnRecognize;
    private TextView tvResult;
    private TextView tvModelInfo;
    private Spinner spinnerModel;

    // ========== 模型相关 ==========
    private Bitmap selectedBitmap;
    private Interpreter currentInterpreter;
    private ActivityResultLauncher<Intent> imagePickerLauncher;

    // ========== 模型配置 ==========
    private static class ModelConfig {
        String tfliteFile;
        String displayName;
        String[] labelsEN;
        String[] labelsCN;
        int imageSize;
        boolean isGrayscale;
        String emoji;
        String description;

        ModelConfig(String tfliteFile, String displayName, String[] labelsEN,
                    String[] labelsCN, int imageSize, boolean isGrayscale,
                    String emoji, String description) {
            this.tfliteFile = tfliteFile;
            this.displayName = displayName;
            this.labelsEN = labelsEN;
            this.labelsCN = labelsCN;
            this.imageSize = imageSize;
            this.isGrayscale = isGrayscale;
            this.emoji = emoji;
            this.description = description;
        }
    }

    private Map<String, ModelConfig> modelConfigs;
    private String currentModelKey = "pneumonia";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initModelConfigs();
        initializeViews();
        setupSpinner();
        setupImagePicker();
        setupButtons();
        loadModel(currentModelKey);
    }

    /**
     * 初始化所有模型配置
     */
    private void initModelConfigs() {
        modelConfigs = new HashMap<>();

        // 肺炎检测
        modelConfigs.put("pneumonia", new ModelConfig(
                "pneumonia_model.tflite",
                "肺炎检测",
                new String[]{"NORMAL", "PNEUMONIA"},
                new String[]{"正常", "肺炎"},
                224,
                true,  // 灰度图
                "🫁",
                "X光胸片肺炎诊断"
        ));

        // 乳腺癌检测
        modelConfigs.put("breast", new ModelConfig(
                "breast_model.tflite",
                "乳腺癌检测",
                new String[]{"BENIGN", "MALIGNANT"},
                new String[]{"良性", "恶性"},
                224,
                false,  // RGB图像
                "🎀",
                "超声图像乳腺癌筛查"
        ));

        // 脑肿瘤检测
        modelConfigs.put("brain", new ModelConfig(
                "brain_model.tflite",
                "脑肿瘤检测",
                new String[]{"NO_TUMOR", "TUMOR"},
                new String[]{"无肿瘤", "有肿瘤"},
                224,
                true,  // 灰度图
                "🧠",
                "MRI脑部肿瘤诊断"
        ));

        // 疟疾检测 - 按字母顺序：Parasitized在前，Uninfected在后
        modelConfigs.put("malaria", new ModelConfig(
                "malaria_model.tflite",
                "疟疾检测",
                new String[]{"PARASITIZED", "UNINFECTED"},  // 修正顺序！
                new String[]{"感染", "未感染"},              // 修正顺序！
                150,
                false,  // RGB图像
                "🦟",
                "血液细胞疟疾筛查"
        ));
    }

    private void initializeViews() {
        imageView = findViewById(R.id.imageView);
        btnSelect = findViewById(R.id.btnSelect);
        btnRecognize = findViewById(R.id.btnRecognize);
        tvResult = findViewById(R.id.tvResult);
        tvModelInfo = findViewById(R.id.tvModelInfo);
        spinnerModel = findViewById(R.id.spinnerModel);
    }

    /**
     * 设置模型选择下拉框
     */
    private void setupSpinner() {
        String[] modelNames = {
                "🫁 肺炎检测",
                "🎀 乳腺癌检测",
                "🧠 脑肿瘤检测",
                "🦟 疟疾检测"
        };

        String[] modelKeys = {"pneumonia", "breast", "brain", "malaria"};

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                modelNames
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerModel.setAdapter(adapter);

        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String newModelKey = modelKeys[position];
                if (!newModelKey.equals(currentModelKey)) {
                    currentModelKey = newModelKey;
                    loadModel(currentModelKey);
                    // 清空之前的结果
                    tvResult.setText("请选择图像进行识别");
                    btnRecognize.setEnabled(selectedBitmap != null);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    /**
     * 加载指定模型
     */
    private void loadModel(String modelKey) {
        ModelConfig config = modelConfigs.get(modelKey);
        if (config == null) {
            Toast.makeText(this, "模型配置不存在", Toast.LENGTH_SHORT).show();
            return;
        }

        // 关闭旧模型
        if (currentInterpreter != null) {
            currentInterpreter.close();
            currentInterpreter = null;
        }

        try {
            currentInterpreter = new Interpreter(loadModelFile(config.tfliteFile));

            // 更新模型信息显示
            String info = String.format("%s %s\n%s\n输入: %dx%d %s",
                    config.emoji,
                    config.displayName,
                    config.description,
                    config.imageSize,
                    config.imageSize,
                    config.isGrayscale ? "灰度" : "彩色"
            );
            tvModelInfo.setText(info);

            Toast.makeText(this, "✓ " + config.displayName + " 已加载", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            e.printStackTrace();
            tvModelInfo.setText("❌ 模型加载失败: " + config.tfliteFile);
            Toast.makeText(this, "模型加载失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private MappedByteBuffer loadModelFile(String filename) throws Exception {
        var fileDescriptor = getAssets().openFd(filename);
        var inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        var fileChannel = inputStream.getChannel();
        return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.getStartOffset(),
                fileDescriptor.getDeclaredLength()
        );
    }

    private void setupImagePicker() {
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        handleImageSelected(result.getData().getData());
                    }
                }
        );
    }

    private void handleImageSelected(Uri imageUri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            selectedBitmap = BitmapFactory.decodeStream(inputStream);
            imageView.setImageBitmap(selectedBitmap);
            btnRecognize.setEnabled(currentInterpreter != null);
            tvResult.setText("图片已加载，点击识别按钮开始诊断");
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "图片加载失败", Toast.LENGTH_SHORT).show();
        }
    }

    private void setupButtons() {
        btnSelect.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            imagePickerLauncher.launch(intent);
        });

        btnRecognize.setOnClickListener(v -> {
            if (selectedBitmap != null && currentInterpreter != null) {
                performRecognition();
            }
        });
    }

    private void performRecognition() {
        ModelConfig config = modelConfigs.get(currentModelKey);
        if (config == null) return;

        tvResult.setText("识别中，请稍候...");
        btnRecognize.setEnabled(false);

        new Thread(() -> {
            long startTime = System.currentTimeMillis();
            String result = recognizeImage(selectedBitmap, config);
            long endTime = System.currentTimeMillis();

            String finalResult = result + String.format("\n\n⏱ 推理时间: %d ms", endTime - startTime);

            runOnUiThread(() -> {
                tvResult.setText(finalResult);
                btnRecognize.setEnabled(true);
            });
        }).start();
    }

    /**
     * 图像识别核心方法
     */
    private String recognizeImage(Bitmap bitmap, ModelConfig config) {
        try {
            int size = config.imageSize;

            // 1. 调整大小
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, size, size, true);

            // 2. 准备输入buffer
            ByteBuffer inputBuffer;

            if (config.isGrayscale) {
                // 灰度图: [1, size, size, 1]
                inputBuffer = ByteBuffer.allocateDirect(4 * size * size * 1);
                inputBuffer.order(ByteOrder.nativeOrder());

                int[] pixels = new int[size * size];
                resized.getPixels(pixels, 0, size, 0, 0, size, size);

                for (int pixel : pixels) {
                    int r = (pixel >> 16) & 0xFF;
                    int g = (pixel >> 8) & 0xFF;
                    int b = pixel & 0xFF;
                    float gray = (r + g + b) / 3.0f / 255.0f;
                    inputBuffer.putFloat(gray);
                }
            } else {
                // RGB图: [1, size, size, 3]
                inputBuffer = ByteBuffer.allocateDirect(4 * size * size * 3);
                inputBuffer.order(ByteOrder.nativeOrder());

                int[] pixels = new int[size * size];
                resized.getPixels(pixels, 0, size, 0, 0, size, size);

                for (int pixel : pixels) {
                    float r = ((pixel >> 16) & 0xFF) / 255.0f;
                    float g = ((pixel >> 8) & 0xFF) / 255.0f;
                    float b = (pixel & 0xFF) / 255.0f;
                    inputBuffer.putFloat(r);
                    inputBuffer.putFloat(g);
                    inputBuffer.putFloat(b);
                }
            }

            // 3. 准备输出
            float[][] output = new float[1][1];

            // 4. 运行推理
            currentInterpreter.run(inputBuffer, output);

            // 5. 处理结果
            float probability = output[0][0];
            int predictedClass = probability > 0.5 ? 1 : 0;
            float confidence = predictedClass == 1 ? probability * 100 : (1 - probability) * 100;

            // 6. 格式化结果
            String diagnosis = config.labelsCN[predictedClass];
            String diagnosisEN = config.labelsEN[predictedClass];

            // 根据模型类型判断是否异常
            boolean isAbnormal = false;
            switch (currentModelKey) {
                case "pneumonia":
                    isAbnormal = predictedClass == 1; // PNEUMONIA是异常(类别1)
                    break;
                case "breast":
                    isAbnormal = predictedClass == 1; // MALIGNANT是异常(类别1)
                    break;
                case "brain":
                    isAbnormal = predictedClass == 1; // TUMOR是异常(类别1)
                    break;
                case "malaria":
                    isAbnormal = predictedClass == 0; // PARASITIZED是异常(类别0)！
                    break;
            }

            String statusEmoji = isAbnormal ? "⚠️" : "✓";
            String statusText = isAbnormal ? "需要关注" : "正常";

            return String.format(
                    "%s %s 诊断结果\n\n" +
                            "━━━━━━━━━━━━━━━━━━\n" +
                            "诊断: %s\n" +
                            "英文: %s\n" +
                            "置信度: %.2f%%\n" +
                            "━━━━━━━━━━━━━━━━━━\n" +
                            "%s 状态: %s",
                    config.emoji,
                    config.displayName,
                    diagnosis,
                    diagnosisEN,
                    confidence,
                    statusEmoji,
                    statusText
            );

        } catch (Exception e) {
            e.printStackTrace();
            return "❌ 识别失败\n\n" + e.getMessage();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (currentInterpreter != null) {
            currentInterpreter.close();
        }
    }
}
