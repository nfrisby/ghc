{-# LANGUAGE TopLevelKindSignatures #-}
{-# LANGUAGE TypeFamilies, GADTs, DataKinds, ExplicitForAll #-}

-- See also: tlks007_fail.hs
module TLKS_007 where

import Data.Kind (Type, Constraint)

type family F a where { F Type = True; F _ = False }
type family G a where { G Type = False; G _ = True }

type X :: forall k1 k2. (F k1 ~ G k2) => k1 -> k2 -> Type
data X a b where
  MkX :: X Integer Maybe   -- OK: F Type ~ G (Type -> Type)
                           --     True ~ True
