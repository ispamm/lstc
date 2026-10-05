import ast
from dataclasses import replace
import json
from pathlib import Path
import unittest
from loster2026.config import (Config, VARIANTS, PILOT_DATASETS, PILOT_SEEDS,
                              planned_runs, validate_manifest, assert_pilot_contract, load_campaign)
from loster2026.protocol import temperature, changed_fraction, should_stop
from loster2026.selection import summarize
from support import ROOT, REPO

class ContractTests(unittest.TestCase):
    def test_manifest_exact120_paired(self):
        manifest = json.loads((ROOT/"configs/pilot/manifest.json").read_text())
        runs = validate_manifest(manifest)
        self.assertEqual(len(runs), 120)
        for variant in VARIANTS:
            pairs = {(r["dataset"],r["seed"]) for r in runs if r["variant"] == variant}
            self.assertEqual(pairs,{(d,s) for d in PILOT_DATASETS for s in PILOT_SEEDS})

    def test_manifest_rejects_duplicate_and_omission(self):
        for runs in (planned_runs()[:-1], planned_runs()[:-1]+[planned_runs()[0]]):
            with self.assertRaises(ValueError): validate_manifest({"runs":runs})

    def test_configs_identical_except_variant(self):
        values=[]
        for name in ("legacy-clean","no-resoftmax","aligned-init"):
            campaign, config=load_campaign(ROOT/"configs/pilot"/(name+".json"))
            assert_pilot_contract(config)
            value=config.to_dict();value.pop("variant");values.append(value)
            self.assertEqual(campaign["datasets"],list(PILOT_DATASETS))
            self.assertEqual(campaign["seeds"],list(PILOT_SEEDS))
        self.assertEqual(values[0],values[1]);self.assertEqual(values[0],values[2])

    def test_modified_pilot_contract_rejected(self):
        with self.assertRaises(ValueError):assert_pilot_contract(replace(Config(),alpha=2))

    def test_variant_only_excluded_from_cache_id(self):
        base=Config()
        self.assertEqual(base.initialization_id(),replace(base,variant=VARIANTS[1]).initialization_id())
        self.assertNotEqual(base.initialization_id(),replace(base,seed=1).initialization_id())

    def test_final44_and36_exact_from_spec(self):
        final=json.loads((ROOT/"configs/final/benchmark.json").read_text())
        self.assertEqual(len(set(final["core40"])),40)
        self.assertEqual(set(final["extended44"])-set(final["core40"]),
                         {"CinCECGTorso","StarLightCurves","MixedShapesRegularTrain","MixedShapesSmallTrain"})
        self.assertEqual(len(final["primary36"]),36)
        self.assertEqual(set(final["primary36"]),set(final["extended44"])-set(PILOT_DATASETS))

    def test_temperature_schedule(self):
        c=Config()
        for epoch in (0,1,4,9,20,99):
            self.assertEqual(temperature(c,epoch),max(10*.65**epoch,.01))
        self.assertEqual(temperature(c,99),.01)
        with self.assertRaises(ValueError):temperature(c,-1)

    def test_stopping_strict_and_second_epoch(self):
        c=Config()
        self.assertFalse(should_stop([1,1],[1,1],1,c))
        self.assertTrue(should_stop([1,1],[1,1],2,c))
        current=[0]*999+[1];previous=[0]*1000
        self.assertEqual(changed_fraction(current,previous),.001)
        self.assertFalse(should_stop(current,previous,2,c))
        self.assertIsNone(changed_fraction([1],None))
        with self.assertRaises(ValueError):changed_fraction([1],[1,2])

    def test_python_sources_parse_without_execution(self):
        paths=list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").rglob("*.py"))+list((ROOT/"tests").rglob("*.py"))
        self.assertGreater(len(paths),20)
        for path in paths:ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

class SelectionTests(unittest.TestCase):
    def records(self, gain=0, gain_aligned=0):
        records=[]
        for r in planned_runs():
            shift=gain if r["variant"]==VARIANTS[1] else gain_aligned if r["variant"]==VARIANTS[2] else 0
            value=.4+.01*r["seed"]+shift
            records.append({**r,"complete":True,"finite":True,
                            "metrics":{"ARI":value,"NMI_arithmetic":.5,"RI":.6,"ACC":.7},
                            "collapse_original":False,"collapse_augmented":False,"fit_seconds":1.})
        return records

    def test_no_gain_retains_legacy(self):
        self.assertEqual(summarize(self.records())["numerical_recommendation"],VARIANTS[0])

    def test_tiny_gain_is_insufficient(self):
        self.assertEqual(summarize(self.records(.005))["numerical_recommendation"],VARIANTS[0])

    def test_broad_gain_selects_noresoftmax(self):
        report=summarize(self.records(.02))
        self.assertEqual(report["numerical_recommendation"],VARIANTS[1])
        self.assertFalse(report["method_automatically_frozen"])

    def test_tie_simplicity_and_cost_priority(self):
        rows=self.records(.02,.02)
        self.assertEqual(summarize(rows)["numerical_recommendation"],VARIANTS[1])
        for row in rows:
            if row["variant"]==VARIANTS[2]:row["fit_seconds"]=.9
        self.assertEqual(summarize(rows)["numerical_recommendation"],VARIANTS[2])

    def test_new_collapse_disqualifies_gain(self):
        rows=self.records(.2)
        next(r for r in rows if r["variant"]==VARIANTS[1])["collapse_original"]=True
        self.assertEqual(summarize(rows)["numerical_recommendation"],VARIANTS[0])

    def test_failed_legacy_prevents_decision(self):
        rows=self.records(.2);rows[0]["finite"]=False
        self.assertEqual(summarize(rows)["status"],"INVALID_LEGACY_CONTROL")
        self.assertIsNone(summarize(rows)["numerical_recommendation"])

    def test_missing_runs_and_duplicates(self):
        rows=self.records()
        self.assertEqual(summarize(rows[:-1])["status"],"INCOMPLETE")
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])

    def test_sample_sd_ddof1(self):
        report=summarize(self.records())
        sd=report["legacy"]["datasets"][PILOT_DATASETS[0]]["metrics"]["ARI"]["sample_sd_ddof1"]
        self.assertAlmostEqual(sd,(.001/4)**.5,14)
