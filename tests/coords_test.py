import unittest
import torch
import math

from spherical_inr.coords import (
    rtp_to_r3,
    tp_to_r3,
    r3_to_rtp,
    r3_to_tp,
    rt_to_r2,
    t_to_r2,
    r2_to_rt,
    r2_to_t,
)


class TestSphericalToCartesian(unittest.TestCase):

    # === existing tests ===

    def test_rtp_to_r3_valid(self):
        coords = torch.tensor([1.0, math.pi / 2, 0.0])
        out = rtp_to_r3(coords)
        exp = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(out, exp, atol=1e-5))

        batch = torch.tensor([[1.0, math.pi / 2, 0.0], [2.0, math.pi / 2, math.pi]])
        out_b = rtp_to_r3(batch)
        exp_b = torch.tensor([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(out_b, exp_b, atol=1e-5))

    def test_rtp_to_r3_invalid(self):
        with self.assertRaises(ValueError):
            rtp_to_r3(torch.tensor([1.0, math.pi / 2]))

    def test_tp_to_r3_valid(self):
        coords = torch.tensor([math.pi / 2, 0.0])
        out = tp_to_r3(coords)
        exp = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(out, exp, atol=1e-5))

        batch = torch.tensor([[math.pi / 2, 0.0], [math.pi / 2, math.pi]])
        out_b = tp_to_r3(batch)
        exp_b = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(out_b, exp_b, atol=1e-5))

    def test_tp_to_r3_invalid(self):
        with self.assertRaises(ValueError):
            tp_to_r3(torch.tensor([0.0, math.pi / 2, 0.0]))

    def test_rt_to_r2_valid(self):
        coords = torch.tensor([2.0, 0.0])
        out = rt_to_r2(coords)
        exp = torch.tensor([2.0, 0.0])
        self.assertTrue(torch.allclose(out, exp, atol=1e-5))

        batch = torch.tensor([[2.0, 0.0], [3.0, math.pi / 2]])
        out_b = rt_to_r2(batch)
        exp_b = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        self.assertTrue(torch.allclose(out_b, exp_b, atol=1e-5))

    def test_rt_to_r2_invalid(self):
        with self.assertRaises(ValueError):
            rt_to_r2(torch.tensor([1.0]))

    def test_t_to_r2_valid(self):
        coords = torch.tensor([[math.pi / 2]])
        out = t_to_r2(coords)
        exp = torch.tensor([0.0, 1.0])
        self.assertTrue(torch.allclose(out, exp, atol=1e-5))

        batch = torch.tensor([[0.0], [math.pi / 2]])
        out_b = t_to_r2(batch)
        exp_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(out_b, exp_b, atol=1e-5))

    def test_t_to_r2_invalid(self):
        with self.assertRaises(ValueError):
            t_to_r2(torch.tensor([[0.0, 1.0]]))


class TestCartesianToSpherical(unittest.TestCase):

    # r3_to_rtp: full spherical
    def test_r3_to_rtp_valid(self):
        # Round-trip check
        orig = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
        rtp = r3_to_rtp(orig)
        rec = rtp_to_r3(rtp)
        self.assertTrue(torch.allclose(rec, orig, atol=1e-5))

    def test_r3_to_rtp_invalid(self):
        with self.assertRaises(ValueError):
            r3_to_rtp(torch.tensor([1.0, 2.0]))  # missing one coord

    # r3_to_tp: on unit sphere
    def test_r3_to_tp_valid(self):
        # (1,0,0) -> θ=π/2, φ=0
        coord = torch.tensor([1.0, 0.0, 0.0])
        tp = r3_to_tp(coord)
        self.assertAlmostEqual(tp[0].item(), math.pi / 2, places=5)
        self.assertAlmostEqual(tp[1].item(), 0.0, places=5)

        # batch
        batch = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        tp_b = r3_to_tp(batch)
        # (0,0,1) -> θ=0, φ=0 ; (0,1,0) -> θ=π/2, φ=π/2
        self.assertTrue(
            torch.allclose(
                tp_b, torch.tensor([[0.0, 0.0], [math.pi / 2, math.pi / 2]]), atol=1e-5
            )
        )

    def test_r3_to_tp_invalid(self):
        with self.assertRaises(ValueError):
            r3_to_tp(torch.tensor([1.0, 0.0]))


class TestCartesianToPolar2D(unittest.TestCase):

    # r2_to_rt: full polar
    def test_r2_to_rt_valid(self):
        coords = torch.tensor([2.0, 0.0])
        out = r2_to_rt(coords)
        self.assertAlmostEqual(out[0].item(), 2.0, places=5)
        self.assertAlmostEqual(out[1].item(), 0.0, places=5)

        batch = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        out_b = r2_to_rt(batch)
        # (0,2)-> r=2, θ=π/2 ; (1,1)-> r=√2, θ=π/4
        self.assertTrue(
            torch.allclose(
                out_b,
                torch.tensor([[2.0, math.pi / 2], [math.sqrt(2), math.pi / 4]]),
                atol=1e-5,
            )
        )

    def test_r2_to_rt_invalid(self):
        with self.assertRaises(ValueError):
            r2_to_rt(torch.tensor([1.0, 2.0, 3.0]))

    # r2_to_t: unit circle
    def test_r2_to_t_valid(self):
        coords = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = r2_to_t(coords)
        # (1,0)->0 ; (0,1)->π/2
        self.assertTrue(
            torch.allclose(out.squeeze(-1), torch.tensor([0.0, math.pi / 2]), atol=1e-5)
        )

    def test_r2_to_t_invalid(self):
        with self.assertRaises(ValueError):
            r2_to_t(torch.tensor([[1.0], [0.0]]))  # wrong last dim


if __name__ == "__main__":
    unittest.main()
