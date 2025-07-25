// main03.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: basic usage; process selection; command file; python; matplotlib;

// This is a simple test program.
// It illustrates how different processes can be selected and studied.
// All input is specified in the main03.cmnd file.
// Also illustrated output to be plotted by Python/Matplotlib/pyplot.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

//==========================================================================

int main() {

  Pythia8ToHepMC toHepMC("events.hepmc");

  // Generator.
  Pythia pythia;

  // Shorthand for the event record in pythia.
  Event& event = pythia.event;

  // Read in commands from external file.
  pythia.readFile("py8.dat");

  // Extract settings to be used in the main program.
  int nEvent = pythia.mode("Main:numberOfEvents");
  int nAbort = pythia.mode("Main:timesAllowErrors");

  // Initialize.
  pythia.init();

  // Begin event loop.
  int iAbort = 0;
  int nEventWrite = 0;
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Generate events. Quit if many failures.
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // Event filter
    bool pass = false;
    float genHT = 0;
    for (int i = 0; i < pythia.event.size(); ++i) {
      if (pythia.event[i].isFinal()) {
        genHT += pythia.event[i].pT();
        // cout << i << " " << pythia.event[i].status() << " pT = " << pythia.event[i].pT()
        //      << " GeV/c and eta = " << pythia.event[i].eta() << endl;
      }
    }
    if (genHT > 1000) pass = true;

    // Write to HepMC
    pass = true;
    if (pass) {
      ++nEventWrite;
      toHepMC.writeNextEvent( pythia );
    }
  // End of event loop.
  }

  // Final statistics. Normalize and output histograms.
  pythia.stat();
  cout << "Number of events written: " << nEventWrite << endl;

  // Done.
  return 0;
}
