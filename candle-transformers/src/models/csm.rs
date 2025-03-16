//! Implementation of the Conversational Speech Model (CSM) from Sesame
//!
//! See: [SesameAILabs/csm](https://github.com/SesameAILabs/csm)
//!
use super::llama::{Cache as LlamaCache, Config as LlamaConfig, Llama};
use candle::{IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_b, Embedding, Linear, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub enum LlamaFlavor {
    #[serde(rename = "llama-1B")]
    V3_2_1B,
    #[serde(rename = "llama-100M")]
    V3_2_100M,
}

impl LlamaFlavor {
    fn llama_config(&self) -> LlamaConfig {
        match self {
            Self::V3_2_1B => LlamaConfig {
                hidden_size: 2048,
                intermediate_size: 8192,
                vocab_size: 128256,
                num_hidden_layers: 16,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                use_flash_attn: false,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.,
                bos_token_id: None,
                eos_token_id: None,
                rope_scaling: Some(super::llama::Llama3RopeConfig {
                    factor: 32.0,
                    low_freq_factor: 1.0,
                    high_freq_factor: 4.0,
                    original_max_position_embeddings: 8192,
                    rope_type: super::llama::Llama3RopeType::Llama3,
                }),
                max_position_embeddings: 131072,
                tie_word_embeddings: true,
            },
            Self::V3_2_100M => LlamaConfig {
                hidden_size: 1024,
                intermediate_size: 8192,
                vocab_size: 128256,
                num_hidden_layers: 4,
                num_attention_heads: 8,
                num_key_value_heads: 2,
                use_flash_attn: false,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.,
                bos_token_id: None,
                eos_token_id: None,
                rope_scaling: Some(super::llama::Llama3RopeConfig {
                    factor: 32.0,
                    low_freq_factor: 1.0,
                    high_freq_factor: 4.0,
                    original_max_position_embeddings: 8192,
                    rope_type: super::llama::Llama3RopeType::Llama3,
                }),
                max_position_embeddings: 2048,
                tie_word_embeddings: true,
            },
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub backbone_flavor: LlamaFlavor,
    pub decoder_flavor: LlamaFlavor,
    pub text_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_num_codebooks: usize,
}

#[derive(Debug, Clone)]
pub struct Model {
    backbone: Llama,
    decoder: Llama,
    backbone_cache: LlamaCache,
    decoder_cache: LlamaCache,
    text_embeddings: Embedding,
    audio_embeddings: Embedding,
    projection: Linear,
    codebook0_head: Linear,
    audio_head: Tensor,
    config: Config,
}

impl Model {
    pub fn new(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let backbone_cfg = cfg.backbone_flavor.llama_config();
        let decoder_cfg = cfg.decoder_flavor.llama_config();
        let backbone_dim = backbone_cfg.hidden_size;
        let decoder_dim = decoder_cfg.hidden_size;
        let backbone = Llama::load(vb.pp("backbone"), &backbone_cfg)?;
        let decoder = Llama::load(vb.pp("decoder"), &decoder_cfg)?;
        let backbone_cache = LlamaCache::new(true, vb.dtype(), &backbone_cfg, vb.device())?;
        let decoder_cache = LlamaCache::new(true, vb.dtype(), &decoder_cfg, vb.device())?;
        let text_embeddings =
            embedding(cfg.text_vocab_size, backbone_dim, vb.pp("text_embeddings"))?;
        let audio_embeddings = embedding(
            cfg.audio_vocab_size * cfg.audio_num_codebooks,
            backbone_dim,
            vb.pp("audio_embeddings"),
        )?;
        let projection = linear_b(backbone_dim, decoder_dim, false, vb.pp("projection"))?;
        let codebook0_head = linear_b(
            backbone_dim,
            cfg.audio_vocab_size,
            false,
            vb.pp("codebook0_head"),
        )?;
        let audio_head = vb.get(
            (
                cfg.audio_num_codebooks - 1,
                decoder_dim,
                cfg.audio_vocab_size,
            ),
            "audio_head",
        )?;
        Ok(Self {
            backbone,
            decoder,
            backbone_cache,
            decoder_cache,
            text_embeddings,
            audio_head,
            audio_embeddings,
            projection,
            codebook0_head,
            config: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}
