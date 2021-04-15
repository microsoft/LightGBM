Recommendations When Using gcc
==============================

It is recommended to use ``-O3 -mtune=native`` to achieve maximum speed during LightGBM training.

Using Intel Ivy Bridge CPU on 1M x 1K Bosch dataset, the performance increases as follow:

+-------------------------------------+---------------------+
| Compilation Flag                    | Performance Index   |
+=====================================+=====================+
| ``-O2 -mtune=core2``                | 100.00%             |
+-------------------------------------+---------------------+
| ``-O2 -mtune=native``               | 100.90%             |
+-------------------------------------+---------------------+
| ``-O3 -mtune=native``               | 102.78%             |
+-------------------------------------+---------------------+
| ``-O3 -ffast-math -mtune=native``   | 100.64%             |
+-------------------------------------+---------------------+

You can find more details on the experimentation below:

-  `Laurae++/Benchmarks <https://sites.google.com/view/lauraepp/benchmarks/xgb-vs-lgb-feb-2017>`__

-  `Laurae2/gbt\_benchmarks <https://github.com/Laurae2/gbt_benchmarks>`__

-  `Laurae's Benchmark Master Data (Interactive) <https://public.tableau.com/views/gbt_benchmarks/Master-Data?:showVizHome=no>`__

-  `Kaggle Paris Meetup #12 Slides <https://drive.google.com/file/d/0B6qJBmoIxFe0ZHNCOXdoRWMxUm8/view>`__

Some explanatory pictures:

.. image:: ./_static/images/gcc-table.png
   :align: center
   :target: ./_static/images/gcc-table.png
   :alt: table comparing training runtime for different combinations of max depth, compiler flags, and number of threads. Faster training times are shown in green, slower times in red. For max depth 10 and 12, the fastest training was achieved with 5 threads and compiler flag dash O 2.

.. image:: ./_static/images/gcc-bars.png
   :align: center
   :target: ./_static/images/gcc-bars.png
   :alt: picture of a simple bar chart against running time

.. image:: ./_static/images/gcc-chart.png
   :align: center
   :target: ./_static/images/gcc-chart.png
   :alt: a grid of vertical bar charts comparing run time for different combinations of max depth, compiler flags, and number of threads. The charts show that for shallow trees, using more threads is always expected to provide some reduction in run time. But for deeper trees (max depth greater than 10), using a value of num threads that is too high can actually result in slower training.

.. image:: ./_static/images/gcc-comparison-1.png
   :align: center
   :target: ./_static/images/gcc-comparison-1.png
   :alt: a horizontal bar chart comparing Light G B M performance versus compilation flags. For most settings of max depth, best performance was achieved with flags dash O 3, dash M T U N E equals native.

.. image:: ./_static/images/gcc-comparison-2.png
   :align: center
   :target: ./_static/images/gcc-comparison-2.png
   :alt: a set of 4 vertical bar charts comparing Light G B M performance versus compilation flags. For most settings of max depth, best performance was achieved with flags dash O 3, dash M T U N E equals native.

.. image:: ./_static/images/gcc-meetup-1.png
   :align: center
   :target: ./_static/images/gcc-meetup-1.png
   :alt: grid of line charts showing the relative speed of training for different combinations of max depth, number of threads, and compilation flags. The grid shows that for models with max depth greater than 5, compiling Light G B M with default compiler flags produces faster training time than any of the customizations explored.

.. image:: ./_static/images/gcc-meetup-2.png
   :align: center
   :target: ./_static/images/gcc-meetup-2.png
   :alt: comparison of cumulative speed versus the slowest density of each algorithm at various depths with v-2, O-3 remaining constant almost in all cases
