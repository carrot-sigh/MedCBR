import pytest
import torch

from src.models.semantic_feedback import (
    AuxiliaryConceptPrototype,
    SemanticFeedbackPrototype,
)
from src.utils.c3_multilabel import masked_bce_with_logits


@pytest.mark.parametrize("model_class", [AuxiliaryConceptPrototype, SemanticFeedbackPrototype])
def test_hierarchy_prototype_output_contract(model_class):
    model = model_class(dim=16, predictor_hidden=8, num_c1=5, num_c2=43, num_c3=10)
    global_feat = torch.randn(3, 16)
    region_feat = torch.randn(3, 32, 16)
    region_mask = torch.ones(3, 32, dtype=torch.bool)
    region_mask[:, -4:] = False

    output = model(global_feat, region_feat, region_mask)

    assert output["c1_logits"].shape == (3, 32, 5)
    assert output["c2_logits"].shape == (3, 32, 43)
    assert output["c3_logits"].shape == (3, 10)
    assert set(output) == {
        "c1_logits", "c2_logits", "c3_logits", "c1_prob", "c2_prob", "r0", "r1", "r2"
    }


def test_auxiliary_control_has_no_feedback_state_update():
    model = AuxiliaryConceptPrototype(dim=8, predictor_hidden=4)
    region_feat = torch.randn(2, 32, 8)
    output = model(torch.randn(2, 8), region_feat)
    assert output["r1"] is output["r0"]
    assert output["r2"] is output["r0"]


def test_feedback_alpha_starts_conservatively():
    model = SemanticFeedbackPrototype(dim=8, predictor_hidden=4, semantic_hidden=3)
    assert model.c1_feedback.alpha.item() == pytest.approx(0.1)
    assert model.c2_feedback.alpha.item() == pytest.approx(0.1)


@pytest.mark.parametrize("stop_gradient,expects_c1_gradient", [(True, False), (False, True)])
def test_stop_gradient_controls_c3_to_c1_gradient(stop_gradient, expects_c1_gradient):
    model = SemanticFeedbackPrototype(
        dim=8, predictor_hidden=4, semantic_hidden=3, stop_gradient=stop_gradient, dropout=0.0
    )
    output = model(torch.randn(2, 8), torch.randn(2, 32, 8))
    output["c3_logits"].sum().backward()
    c1_grads = [parameter.grad for parameter in model.c1_head.parameters()]
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in c1_grads) is expects_c1_gradient


def test_masked_bce_supports_all_hierarchy_shapes_and_ignores_unknown():
    for shape in ((2, 32, 5), (2, 32, 43), (2, 10)):
        logits = torch.zeros(shape, requires_grad=True)
        target = torch.full(shape, -1, dtype=torch.int8)
        mask = torch.zeros(shape, dtype=torch.bool)
        target.reshape(-1)[0] = 1
        mask.reshape(-1)[0] = True
        loss = masked_bce_with_logits(logits, target, mask)
        assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
        loss.backward()
        assert torch.count_nonzero(logits.grad).item() == 1


def test_region_mask_shape_is_validated():
    model = SemanticFeedbackPrototype(dim=8, predictor_hidden=4, semantic_hidden=3)
    with pytest.raises(ValueError, match="region_mask"):
        model(torch.randn(2, 8), torch.randn(2, 32, 8), torch.ones(2, 31, dtype=torch.bool))
