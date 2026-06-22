# Tinker-Compatible Client - Multimodal Training with Twinkle Dataset Components
#
# This script demonstrates how to reuse Twinkle's dataset components (LazyDataset,
# Preprocessor, Template, DataLoader) with Tinker client for multimodal training.
#
# Key feature: Uses `input_feature_to_datum` to convert Twinkle's InputFeature
# to Tinker's Datum format automatically.
#
# Supported models: Qwen3.5-VL series (e.g., Qwen/Qwen3.5-4B)

import dotenv
import os
from tqdm import tqdm

dotenv.load_dotenv('.env')

# =============================================================================
# Step 1: Initialize Tinker client (MUST be done before importing ServiceClient)
# =============================================================================
from twinkle import init_tinker_client

init_tinker_client()

from tinker import types, ServiceClient

# =============================================================================
# Step 2: Import Twinkle dataset components and conversion function
# =============================================================================
from twinkle.data_format import Trajectory, Message
from twinkle.preprocessor import Preprocessor
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.dataloader import DataLoader
from twinkle.server.common import input_feature_to_datum  # Key: converts InputFeature -> Datum
from twinkle import get_logger

logger = get_logger()

# =============================================================================
# Step 3: Configuration
# =============================================================================
base_model = 'Qwen/Qwen3.5-4B'  # Multimodal vision-language model
base_url = 'http://localhost:8000'

# =============================================================================
# Step 4: Define Preprocessor (identical to Twinkle version)
# =============================================================================
class LatexOCRProcessor(Preprocessor):
    """LaTeX OCR data preprocessor - converts dataset rows to Trajectory.

    This processor is fully compatible with both Twinkle and Tinker clients.
    """

    def __call__(self, rows):
        """Process a batch of rows.

        Args:
            rows: Dict with column names as keys, lists as values.

        Returns:
            Dict with processed rows.
        """
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        """Convert a single row to Trajectory with image and text.

        Args:
            row: Dict with 'image' (PIL Image) and 'text' (str) fields.

        Returns:
            Trajectory with user message (image + prompt) and assistant response.
        """
        return Trajectory(
            messages=[
                Message(
                    role='user',
                    content='<image>Using LaTeX to perform OCR on the image.',
                    images=[row['image']]  # PIL Image from dataset
                ),
                Message(
                    role='assistant',
                    content=row['text']  # LaTeX text label
                ),
            ]
        )


# =============================================================================
# Step 5: Training function
# =============================================================================
def train():
    """Run multimodal training using Twinkle dataset components with Tinker client."""

    # -------------------------------------------------------------------------
    # 5.1: Initialize Tinker ServiceClient and training client
    # -------------------------------------------------------------------------
    logger.info(f'Connecting to Tinker server at {base_url}')
    service_client = ServiceClient(
        base_url=base_url,
        api_key=os.environ.get('MODELSCOPE_TOKEN', 'EMPTY-TOKEN')
    )

    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=16
    )
    logger.info(f'Created LoRA training client for {base_model}')

    # -------------------------------------------------------------------------
    # 5.2: Use Twinkle's LazyDataset to load data
    # -------------------------------------------------------------------------
    logger.info('Loading LaTeX_OCR dataset...')
    dataset = LazyDataset(
        dataset_meta=DatasetMeta('ms://AI-ModelScope/LaTeX_OCR', data_slice=range(500))
    )
    logger.info(f'Dataset loaded with {len(dataset)} samples')

    # -------------------------------------------------------------------------
    # 5.3: Set multimodal template (handles image tokenization)
    # -------------------------------------------------------------------------
    logger.info(f'Setting template for {base_model}')
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=512)

    # -------------------------------------------------------------------------
    # 5.4: Apply preprocessor (converts rows to Trajectory)
    # -------------------------------------------------------------------------
    logger.info('Applying LatexOCRProcessor...')
    dataset.map(LatexOCRProcessor())

    # -------------------------------------------------------------------------
    # 5.5: Encode dataset (Trajectory -> InputFeature with images)
    # -------------------------------------------------------------------------
    logger.info('Encoding dataset (this may take a while)...')
    dataset.encode(batched=True)
    logger.info('Dataset encoding complete')

    # -------------------------------------------------------------------------
    # 5.6: Use Twinkle's DataLoader for batching
    # -------------------------------------------------------------------------
    batch_size = 4
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    logger.info(f'DataLoader created with batch_size={batch_size}')

    # -------------------------------------------------------------------------
    # 5.7: Training loop
    # -------------------------------------------------------------------------
    num_epochs = 3
    learning_rate = 1e-4
    gradient_accumulation_steps = 2

    logger.info(f'Starting training: {num_epochs} epochs, lr={learning_rate}')

    for epoch in range(num_epochs):
        logger.info(f'\n=== Epoch {epoch + 1}/{num_epochs} ===')

        accumulated_datums = []
        step = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}')):
            # -----------------------------------------------------------------
            # KEY CONVERSION: InputFeature -> Datum
            #
            # batch is List[InputFeature], each containing:
            #   - input_ids: token IDs
            #   - labels: training labels
            #   - attention_mask, position_ids: attention fields
            #   - images: List[PIL.Image] (multimodal field)
            #
            # input_feature_to_datum converts each to Datum with:
            #   - model_input.chunks: [ImageChunk, EncodedTextChunk, ...]
            #   - loss_fn_inputs: {target_tokens, weights}
            # -----------------------------------------------------------------
            datums = [input_feature_to_datum(feature) for feature in batch]
            accumulated_datums.extend(datums)

            # Gradient accumulation: accumulate multiple batches before stepping
            should_step = (
                len(accumulated_datums) >= batch_size * gradient_accumulation_steps
                or batch_idx == len(dataloader) - 1  # Last batch
            )

            if should_step and accumulated_datums:
                # Forward + backward pass
                fwdbwd_future = training_client.forward_backward(
                    accumulated_datums,
                    'cross_entropy'
                )

                # Optimizer step
                optim_future = training_client.optim_step(
                    types.AdamParams(learning_rate=learning_rate)
                )

                # Wait for results
                fwdbwd_result = fwdbwd_future.result()
                optim_result = optim_future.result()

                # Log metrics
                if step % 2 == 0 and fwdbwd_result.loss_fn_outputs:
                    try:
                        logger.info(f'Training Metrics: {optim_result}')
                    except Exception as e:
                        logger.warning(f'Failed to compute loss: {e}')

                # Reset accumulation
                accumulated_datums = []
                step += 1

        # -----------------------------------------------------------------
        # 5.8: Save checkpoint after each epoch
        # -----------------------------------------------------------------
        logger.info(f'Saving checkpoint for epoch {epoch + 1}...')
        save_future = training_client.save_state(f'multimodal-epoch-{epoch + 1}')
        save_result = save_future.result()
        logger.info(f'Saved checkpoint: {save_result.path}')

    logger.info('\nTraining completed!')



# =============================================================================
# Main entry point
# =============================================================================
if __name__ == '__main__':
    train()
