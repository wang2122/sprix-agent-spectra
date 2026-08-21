import numpy as np

from sprix_spectra import AnchorResponse, EvalItem, calibrate_item_bank
from sprix_spectra.math_utils import sigmoid

item = EvalItem("research-01", {"research": 1.0}, difficulty=0.0, discrimination=1.0)
anchors = {
    f"reference-agent-{index}": {"research": float(theta)}
    for index, theta in enumerate(np.linspace(-2.0, 2.0, 41))
}
responses = [
    AnchorResponse(agent_id, item.item_id, float(sigmoid(1.4 * (ability["research"] - 0.35))))
    for agent_id, ability in anchors.items()
]

result = calibrate_item_bank((item,), anchors, responses)
print(result.items[0])
print(result.diagnostics[0])
