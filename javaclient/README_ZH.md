# Twinkle Java Client

Twinkle 训练服务的 Java 17 同步客户端。

## 特性

- 自动创建并维护服务端会话心跳，客户端关闭时自动停止。
- 提供模型训练、LoRA、采样、训练任务、检查点、数据集、DataLoader 与输入处理器 API。
- 使用 Builder 显式配置服务地址、认证令牌、超时与会话策略。

## 引入依赖

发布到 Maven Central 后，可在项目中引入：

```xml
<dependency>
  <groupId>io.github.modelscope</groupId>
  <artifactId>twinkle-client-java</artifactId>
  <version>1.0.0</version>
</dependency>
```

在发布前，可直接将本项目导入 IntelliJ IDEA 作为 Maven 项目。

## 最小示例

```java
import io.github.modelscope.twinkle.TwinkleClient;
import io.github.modelscope.twinkle.model.ModelClient;
import io.github.modelscope.twinkle.types.LoraConfig;
import java.util.Map;

try (TwinkleClient client = TwinkleClient.builder()
        .baseUrl("https://twinkle.example.com")
        .apiKey("your-api-key")
        .build()) {
    if (!client.healthCheck()) {
        throw new IllegalStateException("Twinkle server is unavailable");
    }

    ModelClient model = client.models().open("Qwen/Qwen3.6-27B");
    model.addAdapter("default", new LoraConfig(8, 16, "all-linear", 0.01, "none", null));
    model.setLoss("CrossEntropyLoss");
    model.setOptimizer("Adam", Map.of("lr", 1e-4));
}
```

## 数据加载与训练

```java
import com.google.gson.JsonElement;
import io.github.modelscope.twinkle.processor.DataLoaderClient;
import io.github.modelscope.twinkle.processor.DatasetClient;
import io.github.modelscope.twinkle.types.DatasetKind;
import io.github.modelscope.twinkle.types.DatasetMeta;
import java.util.Map;

DatasetClient dataset = client.processors().dataset(
        DatasetKind.DATASET,
        Map.of("dataset_meta", DatasetMeta.of("ms://your-dataset")));
dataset.setTemplate("Qwen3_5Template", Map.of("model_id", "Qwen/Qwen3.6-27B"));
dataset.encode(false, Map.of("batched", true));

DataLoaderClient loader = client.processors().dataLoader(dataset.processorId(), Map.of("batch_size", 4));
for (JsonElement batch : loader) {
    model.forwardBackward(batch);
    model.clipGradAndStep(1.0, 2);
}
```
## 案例参考
src/test/java/io/github/modelscope/twinkle/MultiModalTrainingTest.java
src/test/java/io/github/modelscope/twinkle/SelfCognitionTest.java
