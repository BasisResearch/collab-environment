import torch

def test_consistency(original_code: bool):
    
    print(f"Testing consistency with original code = {original_code}")

    x = torch.arange(10)**3

    # handle_discrete_data
    v=torch.zeros_like(x)

    v[1:] = torch.diff(x)

    a=torch.zeros_like(x)
    if not original_code:   
        a[1:]=torch.diff(v) # fixed code
    else:
        a[:-1]=torch.diff(v) # original code


    result = torch.stack([x,v,a])
    print(f"Forward (handle_discrete_data) result:\n{result}")

    # inverse: that's how run_gnn_frame is implemented

    vv = torch.zeros_like(x)
    for i in range(1,len(x)):
        vv[i] = vv[i-1] + a[i]
        
    xx = torch.zeros_like(x)
    for i in range(1,len(x)):
        xx[i] = xx[i-1] + vv[i]

    result2 = torch.stack([xx,vv,a])
    print(f"Inverse (run_gnn_frame) result:\n{result2}")

    assert torch.allclose(result, result2)

def test_original_code():
    import pytest

    with pytest.raises(AssertionError):
        test_consistency(original_code=True)
    
def test_fixed_code():
    test_consistency(original_code=False)

if __name__ == "__main__":
    test_original_code()
    print("--------------------------------")
    test_fixed_code()