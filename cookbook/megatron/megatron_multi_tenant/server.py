"""
Multi-Tenant Megatron LoRA Training - Server.

Creates a shared base model and provides APIs for multi-tenant training.
"""

import argparse
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============ Request/Response Models ============

class InitializeRequest(BaseModel):
    lora_config: Optional[Dict[str, Any]] = None
    optimizer_cls: str = "AdamW"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

class InputsRequest(BaseModel):
    inputs: Any

class TenantResponse(BaseModel):
    status: str = "ok"
    tenant_id: Optional[str] = None
    data: Optional[Any] = None


# ============ Server ============

class MultiTenantServer:
    """Server managing multi-tenant Megatron model."""
    
    TIMEOUT = 60 * 30  # 30 min heartbeat timeout
    
    def __init__(self, model_id: str, tp_size: int = 1):
        self.model_id = model_id
        self.tp_size = tp_size
        self.model = None
        self._heartbeats: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def setup(self):
        """Initialize model."""
        from twinkle.megatron.model import (
            MultiTenantMegatronModel,
            initialize_megatron_model,
        )
        
        logger.info(f"Loading model: {self.model_id}")
        base_model, config = initialize_megatron_model(
            model_id=self.model_id,
            tensor_parallel_size=self.tp_size,
        )
        
        # Freeze base model
        for p in base_model.parameters():
            p.requires_grad = False
        
        self.model = MultiTenantMegatronModel(base_model, config)
        logger.info("Server ready")
        
        # Start heartbeat monitor
        threading.Thread(target=self._monitor, daemon=True).start()
    
    def _monitor(self):
        """Cleanup inactive tenants."""
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                expired = [t for t, ts in self._heartbeats.items() if now - ts > self.TIMEOUT]
            for tid in expired:
                logger.warning(f"Tenant {tid} timed out")
                try:
                    self.finalize(tid)
                except:
                    pass
    
    def _heartbeat(self, tenant_id: str):
        with self._lock:
            self._heartbeats[tenant_id] = time.time()
    
    def initialize(self, request: InitializeRequest) -> str:
        """Initialize tenant."""
        from peft import LoraConfig
        
        lora_config = None
        if request.lora_config:
            lora_config = LoraConfig(**request.lora_config)
        
        opt_map = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}
        opt_cls = opt_map.get(request.optimizer_cls, torch.optim.AdamW)
        
        tenant_id = self.model.initialize(
            lora_config=lora_config,
            optimizer_cls=opt_cls,
            optimizer_kwargs=request.optimizer_kwargs,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            max_grad_norm=request.max_grad_norm,
        )
        
        self._heartbeat(tenant_id)
        return tenant_id
    
    def finalize(self, tenant_id: str):
        """Finalize tenant."""
        self.model.finalize(tenant_id)
        with self._lock:
            self._heartbeats.pop(tenant_id, None)
    
    def forward_backward(self, tenant_id: str, inputs: Any) -> Dict:
        """Forward + backward."""
        self._heartbeat(tenant_id)
        
        with self.model.scope(tenant_id):
            output = self.model(inputs)
            # Compute loss (simplified - real impl would depend on task)
            loss = output.mean() if isinstance(output, torch.Tensor) else torch.tensor(0.0)
            self.model.backward(loss)
            return {"loss": loss.item()}
    
    def finish_grad_sync(self, tenant_id: str):
        self._heartbeat(tenant_id)
        self.model.finish_grad_sync(tenant_id)
    
    def clip_grad_norm(self, tenant_id: str) -> float:
        self._heartbeat(tenant_id)
        return self.model.clip_grad_norm(tenant_id=tenant_id).item()
    
    def step(self, tenant_id: str):
        self._heartbeat(tenant_id)
        self.model.step(tenant_id)
    
    def zero_grad(self, tenant_id: str):
        self._heartbeat(tenant_id)
        self.model.zero_grad(tenant_id)
    
    def lr_step(self, tenant_id: str):
        self._heartbeat(tenant_id)
        self.model.lr_step(tenant_id)
    
    def list_tenants(self) -> List[str]:
        return self.model.list_tenants()


# ============ FastAPI App ============

def create_app(server: MultiTenantServer) -> FastAPI:
    """Create FastAPI app."""
    app = FastAPI(title="Multi-Tenant Megatron Server")
    
    def get_tenant(request: Request) -> str:
        tid = request.headers.get("X-Tenant-ID")
        if not tid:
            raise HTTPException(400, "Missing X-Tenant-ID")
        return tid
    
    @app.post("/initialize", response_model=TenantResponse)
    def initialize(body: InitializeRequest):
        tid = server.initialize(body)
        return TenantResponse(tenant_id=tid)
    
    @app.post("/finalize", response_model=TenantResponse)
    def finalize(request: Request):
        server.finalize(get_tenant(request))
        return TenantResponse()
    
    @app.post("/forward_backward", response_model=TenantResponse)
    def forward_backward(request: Request, body: InputsRequest):
        data = server.forward_backward(get_tenant(request), body.inputs)
        return TenantResponse(data=data)
    
    @app.post("/finish_grad_sync", response_model=TenantResponse)
    def finish_grad_sync(request: Request):
        server.finish_grad_sync(get_tenant(request))
        return TenantResponse()
    
    @app.post("/clip_grad_norm", response_model=TenantResponse)
    def clip_grad_norm(request: Request):
        norm = server.clip_grad_norm(get_tenant(request))
        return TenantResponse(data=norm)
    
    @app.post("/step", response_model=TenantResponse)
    def step(request: Request):
        server.step(get_tenant(request))
        return TenantResponse()
    
    @app.post("/zero_grad", response_model=TenantResponse)
    def zero_grad(request: Request):
        server.zero_grad(get_tenant(request))
        return TenantResponse()
    
    @app.post("/lr_step", response_model=TenantResponse)
    def lr_step(request: Request):
        server.lr_step(get_tenant(request))
        return TenantResponse()
    
    @app.get("/tenants")
    def tenants():
        return {"tenants": server.list_tenants()}
    
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    server = MultiTenantServer(args.model_id, args.tp)
    server.setup()
    
    import uvicorn
    uvicorn.run(create_app(server), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
