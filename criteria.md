<!--
What to cluster on, in words. Read by `python customcluster.py`.
Full documentation: README.md -> "Clustering on your own criterion".

Comment: A leading # is read as a section heading, so a #
comment inside # Settings silently discards every setting below it.

CURRENT CORPUS: fileinput/annual2603e.pdf - Nintendo Co., Ltd. Annual Report
2026, 103 pages. The labels below mirror its statutory section order.
-->

# Mode
anchor


# Labels
<!-- `name: description`, one per line. The description is embedded with the
name and carries most of the matching signal, so write it like you would explain
the bucket to a colleague. `front matter` and `financial statements` are here to
absorb navigation furniture and number tables; without them that text spreads
across every real bucket. -->
- front matter: table of contents with page numbers and dot leaders, cover page, translation disclaimer that the Japanese original shall prevail, investor relations contact information, independent auditor, company address
- history and group structure: founded 1947 as a playing card manufacturer in Kyoto, successive changes of company name, stock exchange listings, subsidiaries and associates, consolidated group companies, percentage of voting rights held
- business and strategy: management policy, business model, dedicated video game platform, integrated hardware and software development, Nintendo Switch, digital sales, expanding the number of people who have access to Nintendo IP, mid to long term strategy, issues to address
- financial results: analysis of financial position and operating results by management, net sales, operating profit, ordinary profit, profit attributable to owners of parent, cash flows from operating investing and financing activities, segment performance, key financial data and trends
- risk factors: risks that could adversely affect operating results share price and financial condition, intensifying competition, shifts in consumer preferences, dependence on hit software, foreign exchange rate fluctuation, intellectual property infringement, natural disaster and pandemic
- sustainability and environment: approach to and initiatives for sustainability, sustainability priorities and material topics, climate related disclosure, greenhouse gas emissions, reducing environmental impact of products, supply chain and procurement policy, human rights
- employees: human capital strategy, hiring and development of talent, training, diversity and inclusion, promoting a comfortable working environment, number of employees by segment, average years of service, wage differential between men and women, ratio of female managers
- corporate governance: board of directors and outside directors, audit and supervisory committee, internal control system, remuneration and compensation of officers, policy on cross shareholdings, skills matrix, nomination and appointment of directors, career history of each officer
- shares and dividends: total number of shares authorized and issued, share subscription rights, acquisition and cancellation of treasury shares, dividend policy, dividend per share, major shareholders and shareholding ratio, status of shares by shareholder type
- facilities and R&D: overview of capital investments, major facilities and their book value, plans for new installation and retirement of equipment, research and development activities, underlying technology research, R&D expenditure
- financial statements: consolidated balance sheet, consolidated statement of income and comprehensive income, notes to consolidated financial statements, significant accounting policies, depreciation method, deferred tax assets, millions of yen line items and totals


# Settings
<!-- assign  seeded | hard
     floor   auto (2 sigma below this run's mean) | a cosine | 0 to disable
     model   MUST match the model the corpus was embedded with
     scheme  name stored per row as `cluster_scheme`
     preview member chunks printed per cluster -->
assign = seeded
floor = auto
model = bge-m3
scheme = nintendo-ar2026
preview = 6
