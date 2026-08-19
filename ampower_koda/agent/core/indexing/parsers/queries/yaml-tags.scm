; YAML tags.
;
; A YAML document has no functions and no classes — its structure *is* its
; content. So every mapping key becomes a definition, at every depth, and the
; container resolution that turns `total` into `Cart.total` for Python turns
; `build` into `jobs.build` here for free, from the extents alone.
;
; Keys are given the `field` role, which keeps them addressable by name while
; excluding them from corpus-wide term statistics. That matters more here than
; anywhere else: a repository's CI config and its `hooks` files repeat the same
; two dozen words thousands of times, and letting them vote on document
; frequency would quietly degrade ranking for every other file.

(block_mapping_pair
  key: (flow_node [(plain_scalar (string_scalar) @name)
                   (single_quote_scalar) @name
                   (double_quote_scalar) @name])) @definition.field

(flow_pair
  key: (flow_node [(plain_scalar (string_scalar) @name)
                   (single_quote_scalar) @name
                   (double_quote_scalar) @name])) @definition.field

; Anchors are the one thing in YAML that behaves like a symbol: `&defaults`
; declares a name and `*defaults` uses it.
(anchor (anchor_name) @name) @definition.constant
(alias (alias_name) @name) @reference.type
