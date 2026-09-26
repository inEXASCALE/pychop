Time-series symbols and parameter export
========================================

``SymbolicTimeSeries`` learns an empirical SAX-style representation: normalize
with training-set statistics, average non-overlapping time segments (PAA), and
assign each average to an empirical quantile bin. Symbols are integers in
``[0, alphabet_size)``. This API performs discretization, not symbolic algebra,
equation discovery, forecasting or Gaussian-breakpoint SAX.

Complete fit / transform / reconstruction
-----------------------------------------

.. code-block:: python

   import numpy as np
   from pychop import SymbolicTimeSeries

   t = np.arange(256) * .05
   series = np.sin(t) + .1 * np.cos(3 * t)
   train, test = series[:192], series[192:]
   encoder = SymbolicTimeSeries(segment_size=4, alphabet_size=8)
   train_symbols = encoder.fit_transform(train)
   test_symbols = encoder.transform(test)
   approximation = encoder.inverse_transform(test_symbols)
   print("48 training symbols:", train_symbols.shape)
   print("16 test symbols:", test_symbols)
   print("reconstruction RMSE:", np.sqrt(np.mean((test - approximation)**2)))

``fit`` returns the instance and learns all calibration parameters. ``transform``
never updates them. Split by time before fitting to avoid using future statistics.
For a batch, use shape ``(n_series, n_samples)``; arbitrary leading axes are kept.
Calibration pools all training series into one mean/scale and one set of bins.
For per-channel calibration, fit a separate instance for each channel.

Parameter choices and edge cases
---------------------------------

* ``segment_size``: number of samples averaged into one symbol. Larger values
  remove more temporal detail. The time-axis length must be divisible by this
  value; no samples are silently dropped or padded.
* ``alphabet_size``: 2..256 bins, stored as uint8. More bins retain amplitude
  detail but may need more training data to estimate reliably.
* ``normalize=True``: subtract the global training mean and divide by the
  population standard deviation. Constant training data use scale 1.
* Empirical breakpoints are quantiles of the normalized training segment means.
  A value equal to a breakpoint enters the upper bin (``side="right"``).
  Repeated breakpoints are allowed; constant or discrete data can leave bins empty.
* Reconstruction uses the training mean of each populated bin. Empty bins use
  the corresponding training quantile midpoint as a representative. Repeating
  those centers yields a lossy piecewise-constant time series.
* NaN, infinity, empty training series, complex inputs and invalid lengths raise
  explicit errors. Imputation, resampling and padding are application decisions.
* An unfitted transform raises ``RuntimeError``. A failed fit preserves the prior
  fitted state. Concurrent fitting on the same instance is not supported.

Export and reuse without refitting
----------------------------------

.. code-block:: python

   import json
   from pathlib import Path
   import numpy as np
   from pychop import SymbolicTimeSeries

   train = np.sin(np.arange(128) / 10)
   test = np.sin(np.arange(128, 160) / 10)
   encoder = SymbolicTimeSeries(4, 8).fit(train)
   Path("symbolizer.json").write_text(json.dumps(encoder.to_dict(), indent=2))
   np.save("symbols.npy", encoder.transform(test), allow_pickle=False)

   loaded = SymbolicTimeSeries.from_dict(json.loads(Path("symbolizer.json").read_text()))
   symbols = np.load("symbols.npy", allow_pickle=False)
   np.testing.assert_array_equal(loaded.transform(test), symbols)
   print(loaded.inverse_transform(symbols))

The versioned schema contains ``segment_size``, ``alphabet_size``, ``normalize``,
``mean``, ``scale``, ``breakpoints`` and ``centers``. Loading validates fields,
array lengths, finiteness, positive scale and ordering. Exported arrays are copies,
so editing the mapping does not mutate the fitted object. Store sampling rate,
channel names and time origin alongside the mapping if your application needs
them; those metadata are not inferred by Pychop.

Combine with P3109 and a toy model
----------------------------------

.. code-block:: python

   import numpy as np
   from pychop import P3109, P3109Format, SymbolicTimeSeries

   rng = np.random.default_rng(7)
   series = np.zeros(256)
   for i in range(1, len(series)):
       series[i] = .9 * series[i - 1] + .1 * rng.normal()
   train, test = series[:192], series[192:]
   q = P3109(P3109Format(8, 4), saturate=True)
   encoder = SymbolicTimeSeries(4, 8).fit(q(train))
   symbols = encoder.transform(q(test))
   a = np.dot(train[:-1], train[1:]) / np.dot(train[:-1], train[:-1])
   prediction = q(q(a) * q(np.r_[train[-1], test[:-1]]))
   print("AR(1) coefficient:", a, "quantized:", q(a))
   print("one-step prediction RMSE:", np.sqrt(np.mean((prediction - test)**2)))
   print("symbols:", symbols)

This is one-step forecasting using the observed previous sample, not recursive
multi-step forecasting. Quantization error, forecasting error and symbol
reconstruction error measure different effects. :doc:`examples` combines these
measurements and exports model parameters, policies, integer codes and metrics.

API
---

.. automodule:: pychop.timeseries
   :members:
