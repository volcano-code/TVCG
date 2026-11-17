"""Deterministic persistent control reports; call through bounded_run.py."""
import argparse,json,unittest
from pathlib import Path
import test_validator as tv
import validator

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    reports=[];counter=0
    def persist(self,mode='cavity',mutate=None):
        nonlocal counter
        counter+=1;dest=a.out/f'{counter:03d}-{self._testMethodName}';g,r=tv.fixture(dest,mode)
        if mutate:mutate(dest,g,r)
        rh,sh=tv.save(dest,g,r)
        report=validator.validate(dest/'registry.json',dest/'result.json',registry_sha256=rh,result_sha256=sh)
        (dest/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        reports.append({'test':self.id(),'control':str(self._subtest) if self._subtest else self._testMethodName,'directory':str(dest),'status':report['status'],'gates':report['gates'],'registry_sha256':rh,'result_sha256':sh})
        return report
    tv.ValidatorControls.run_case=persist
    tv.IntegrationControls.run_case=persist
    suite=unittest.TestSuite()
    loader=unittest.TestLoader();suite.addTests(loader.loadTestsFromTestCase(tv.ValidatorControls))
    for name in sorted(tv.IntegrationControls.__dict__):
        if name.startswith('test_'):suite.addTest(tv.IntegrationControls(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'schema':'m90.2.controls.v1','tests_run':result.testsRun,'failures':[(str(x),y) for x,y in result.failures],'errors':[(str(x),y) for x,y in result.errors],'successful':result.wasSuccessful(),'saved_audits':reports,'scope':'analytic engineering fixtures only; no M83 design success','seed':'deterministic no RNG','gpu_used':False,'optimizer_steps':0}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
